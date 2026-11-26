"""Lossless serialization of model parameters, topology, and inference metadata."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import re
import struct
from types import MappingProxyType

from .errors import ValidationError

_MAGIC = b'CSBNDL3\x00'
_MAX_BYTES = 2 * 1024 * 1024
_HEADER_CAP = 65536
_GRAPH_HEADER_CAP = 262144
_ARRAY_DTYPES = {'f2': ('e', 2), 'f4': ('f', 4), 'f8': ('d', 8),
                 'i1': ('b', 1), 'i2': ('h', 2), 'i4': ('i', 4), 'i8': ('q', 8),
                 'u1': ('B', 1), 'u2': ('H', 2), 'u4': ('I', 4), 'u8': ('Q', 8),
                 'bool': ('?', 1)}
_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_.-]{0,127}\Z')


def canonical_json(value):
    """Canonical finite-only UTF-8 JSON, with string mapping keys at every level."""
    def validate(v, depth=0):
        if depth > 64:
            raise ValidationError('metadata nesting limit exceeded')
        if v is None or type(v) in (str, bool, int):
            return
        if type(v) is float:
            if not math.isfinite(v):
                raise ValidationError('metadata numbers must be finite')
            return
        if type(v) is list:
            for item in v:
                validate(item, depth + 1)
            return
        if type(v) is dict and all(type(k) is str for k in v):
            for item in v.values():
                validate(item, depth + 1)
            return
        raise ValidationError('metadata must be plain JSON types')
    validate(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (ValueError, TypeError, UnicodeError, OverflowError) as exc:
        raise ValidationError('invalid metadata JSON') from exc


def _unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValidationError('duplicate JSON key')
        out[key] = value
    return out


@dataclass(frozen=True)
class ArrayPayload:
    """Immutable original dtype/shape and little-endian contiguous raw bytes.

    shape is a tuple of nonnegative integer extents, scalar shape is (). Empty
    extents are explicit, never interpreted as missing. Byte lengths are exact.
    """
    dtype: str
    shape: tuple
    data: bytes

    def __post_init__(self):
        if type(self.dtype) is not str or self.dtype not in _ARRAY_DTYPES:
            raise ValidationError('unsupported array dtype')
        if type(self.shape) is not tuple or len(self.shape) > 32 or any(type(n) is not int or n < 0 for n in self.shape):
            raise ValidationError('shape must be a bounded tuple of nonnegative integer extents')
        if type(self.data) is not bytes:
            raise ValidationError('immutable raw bytes required')
        expected = math.prod(self.shape) * _ARRAY_DTYPES[self.dtype][1]
        if expected != len(self.data):
            raise ValidationError('array shape/dtype byte length mismatch')
        if len(self.data) > _MAX_BYTES:
            raise ValidationError('array exceeds maximum inference bundle budget')


class ModelBundle:
    """Original-dtype inference bundle with independently hashed execution decode.

    kind is actor/vector_q/graph. metadata requires architecture, activations,
    enabled, explicit normalizer+decoder descriptors, execution_dtype and exact
    array_specs {name: {dtype, shape, role}}. Identity is independent of query
    preference. All stochastic/query-dependent execution rules belong in queries.

    The source owner must include every inference-dependent array in array_specs,
    including ensemble members, logstd and nonidentity normalizers when applicable.
    Descriptors may refer to arrays with an ``arrays`` list, checked for presence.
    """

    def __init__(self, kind, arrays, metadata):
        if type(kind) is not str or kind not in ('actor', 'vector_q', 'graph'):
            raise ValidationError('unsupported inference bundle kind')
        if type(arrays) is not dict or not arrays:
            raise ValidationError('nonempty named inference arrays required')
        if any(type(name) is not str or not _NAME.fullmatch(name) or '..' in name for name in arrays):
            raise ValidationError('array names must be simple identifiers, never paths')
        if any(not isinstance(a, ArrayPayload) for a in arrays.values()):
            raise ValidationError('ArrayPayload values required')
        canonical_json(metadata)
        if type(metadata) is not dict:
            raise ValidationError('metadata object required')
        self._kind = kind
        self._arrays = dict(arrays)
        self._metadata = deepcopy(metadata)
        self._validate()
        # Bound headers and data before any container allocation.
        self.encode()

    @property
    def kind(self):
        return self._kind

    @property
    def arrays(self):
        return MappingProxyType(self._arrays)

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    def _validate(self):
        m = self._metadata
        required = {'architecture', 'activations', 'enabled', 'normalizer', 'decoder', 'execution_dtype', 'array_specs'}
        if not required <= m.keys():
            raise ValidationError('missing required inference metadata')
        if type(m['architecture']) is not dict or not m['architecture'] or type(m['architecture'].get('type')) is not str or not m['architecture']['type']:
            raise ValidationError('architecture must explicitly describe its type')
        if type(m['activations']) is not list or not m['activations'] or any(type(a) is not str or not a for a in m['activations']):
            raise ValidationError('explicit activation list required')
        if type(m['enabled']) is not bool:
            raise ValidationError('explicit enabled flag required')
        if m['execution_dtype'] not in ('f2', 'f4', 'f8'):
            raise ValidationError('explicit supported execution float dtype required')
        for component in ('normalizer', 'decoder'):
            d = m[component]
            if type(d) is not dict or type(d.get('type')) is not str or not d['type']:
                raise ValidationError(f'explicit {component} descriptor required')
        specs = m['array_specs']
        if type(specs) is not dict or set(specs) != set(self._arrays):
            raise ValidationError('array_specs must describe every array exactly')
        for name, array in self._arrays.items():
            s = specs[name]
            if type(s) is not dict or set(s) != {'dtype', 'shape', 'role'} or type(s['dtype']) is not str or s['dtype'] != array.dtype or type(s['shape']) is not list or any(type(n) is not int for n in s['shape']) or s['shape'] != list(array.shape):
                raise ValidationError('array dtype/shape/spec mismatch')
            if type(s['role']) is not str or not s['role']:
                raise ValidationError('explicit inference array role required')
        allowed_roles = {'actor': ('weight', 'actor'), 'vector_q': ('weight', 'vector_q'),
                         'graph': ('weight', 'graph_weight')}[self.kind]
        if not any(s['role'] in allowed_roles for s in specs.values()):
            raise ValidationError('inference weight component required')
        def check_refs(value):
            if isinstance(value, dict):
                if 'arrays' in value:
                    refs = value['arrays']
                    if type(refs) is not list or any(type(n) is not str or n not in self._arrays for n in refs):
                        raise ValidationError('inference component references missing arrays')
                for item in value.values():
                    check_refs(item)
            elif isinstance(value, list):
                for item in value:
                    check_refs(item)
        check_refs(m)
        if self.kind == 'vector_q':
            if not {'action_grid', 'argmax', 'tie_rule', 'objective_dim'} <= m.keys():
                raise ValidationError('vector-Q needs grid, scalarization/argmax, tie rule and objective dimension')
            if type(m['action_grid']) is not list or not m['action_grid'] or any(type(a) not in (int,float) for a in m['action_grid']):
                raise ValidationError('nonempty numeric vector-Q action grid required')
            if type(m['objective_dim']) is not int or m['objective_dim'] not in (2, 3):
                raise ValidationError('vector-Q objective_dim must be 2 or 3')
            if m['argmax'] not in ('weighted_sum','envelope_max') or m['tie_rule'] not in ('first','last','random'):
                raise ValidationError('unsupported vector-Q decision/tie rule')
        if self.kind == 'graph':
            if not {'topology', 'node_flags', 'edge_flags', 'inference_state'} <= m.keys():
                raise ValidationError('graph inference needs topology, node flags and inference state')
            t = m['topology']
            if type(t) is not dict or set(t) != {'nodes', 'edges'} or type(t['nodes']) is not list or not t['nodes'] or type(t['edges']) is not list:
                raise ValidationError('explicit graph nodes and edges required')
            if any(type(n) is not int for n in t['nodes']) or len(set(t['nodes'])) != len(t['nodes']):
                raise ValidationError('graph nodes must be unique integer identities')
            if any(type(e) is not list or len(e) != 2 or any(type(n) is not int or n not in t['nodes'] for n in e) for e in t['edges']):
                raise ValidationError('graph edge references unknown node')
            if type(m['node_flags']) is not list or len(m['node_flags']) != len(t['nodes']) or any(type(f) is not bool for f in m['node_flags']):
                raise ValidationError('graph requires per-node enabled flags')
            if type(m['edge_flags']) is not list or len(m['edge_flags']) != len(t['edges']) or any(type(f) is not bool for f in m['edge_flags']):
                raise ValidationError('graph requires per-edge enabled flags')
            if type(m['inference_state']) is not dict or not m['inference_state']:
                raise ValidationError('explicit graph inference state or stateless declaration required')
        self._validate_structure()

    def _validate_structure(self):
        """Validate a finite executable layout grammar, never accept opaque architectures.

        Supported linear/MLP/flat-MLP, multihead concatenation, equal-shape mean
        ensembles and graph forms describe every parameter's semantic placement.
        Unsupported source layouts must extend this grammar and its synthetic tests
        before they can be admitted as complete storage endpoints.
        """
        m, used = self._metadata, set()

        def dim(n):
            if type(n) is not int or not 0 < n <= 2**20:
                raise ValidationError('architecture dimension must be a positive bounded integer')
            return n

        def keys(d, required):
            if type(d) is not dict or set(d) != set(required):
                raise ValidationError('architecture descriptor requires exact structural fields')

        def parameter(name, shape, roles):
            if type(name) is not str or name not in self._arrays:
                raise ValidationError(f'missing inference parameter/bias: {name}')
            a = self._arrays[name]
            if a.shape != tuple(shape) or not a.dtype.startswith('f'):
                raise ValidationError(f'inference parameter shape/dtype mismatch: {name}')
            if m['array_specs'][name]['role'] not in roles:
                raise ValidationError(f'inference parameter role mismatch: {name}')
            used.add(name)
            return a

        def network(d, depth=0):
            if depth > 16 or type(d) is not dict:
                raise ValidationError('architecture nesting limit or malformed descriptor')
            kind = d.get('type')
            if kind == 'identity':
                keys(d, ('type','input_dim'))
                n = dim(d['input_dim'])
                return n, n, 0
            if kind == 'linear':
                keys(d, ('type','input_dim','output_dim','weight','bias'))
                ni, no = dim(d['input_dim']), dim(d['output_dim'])
                parameter(d['weight'], (no,ni), ('weight','actor','vector_q'))
                parameter(d['bias'], (no,), ('bias',))
                return ni, no, 1
            if kind in ('mlp','flat_mlp'):
                required = ('type','layer_sizes','weights','biases') if kind == 'mlp' else ('type','layer_sizes','parameters','parameter_order')
                keys(d, required)
                sizes = d['layer_sizes']
                if type(sizes) is not list or not 2 <= len(sizes) <= 65:
                    raise ValidationError('MLP requires explicit input/hidden/output dimensions')
                for n in sizes:
                    dim(n)
                if kind == 'flat_mlp':
                    if d['parameter_order'] != 'row_major_weight_then_bias':
                        raise ValidationError('unsupported flat MLP parameter order')
                    count = sum(a*b+b for a,b in zip(sizes,sizes[1:]))
                    parameter(d['parameters'], (count,), ('weight','actor','vector_q'))
                else:
                    if type(d['weights']) is not list or type(d['biases']) is not list or len(d['weights']) != len(sizes)-1 or len(d['biases']) != len(sizes)-1:
                        raise ValidationError('MLP requires weights and biases for every layer')
                    for ni,no,w,b in zip(sizes,sizes[1:],d['weights'],d['biases']):
                        parameter(w,(no,ni),('weight','actor','vector_q'))
                        parameter(b,(no,),('bias',))
                return sizes[0], sizes[-1], len(sizes)-1
            if kind == 'multihead':
                keys(d, ('type','trunk','heads','combine'))
                if d['combine'] != 'concatenate' or type(d['heads']) is not list or not d['heads']:
                    raise ValidationError('multihead requires explicit concatenated heads')
                ni,trunk_out,layers = network(d['trunk'],depth+1)
                out = 0
                for h in d['heads']:
                    hi,ho,hl = network(h,depth+1)
                    if hi != trunk_out:
                        raise ValidationError('multihead trunk/head dimension mismatch')
                    out += ho
                    layers += hl
                return ni, out, layers
            if kind == 'ensemble':
                keys(d, ('type','members','combine'))
                if d['combine'] != 'mean' or type(d['members']) is not list or not d['members']:
                    raise ValidationError('ensemble requires explicit mean-combined members')
                dimensions = [network(x,depth+1) for x in d['members']]
                if any(x[:2] != dimensions[0][:2] for x in dimensions):
                    raise ValidationError('ensemble member dimension mismatch')
                return *dimensions[0][:2], sum(x[2] for x in dimensions)
            if kind == 'graph' and self.kind == 'graph':
                keys(d, ('type','input_nodes','output_nodes','weight','bias'))
                nodes,edges = m['topology']['nodes'],m['topology']['edges']
                for name in ('input_nodes','output_nodes'):
                    values = d[name]
                    if type(values) is not list or not values or any(type(n) is not int or n not in nodes for n in values) or len(set(values)) != len(values):
                        raise ValidationError('graph requires valid unique input/output node lists')
                parameter(d['weight'], (len(edges),), ('weight','graph_weight'))
                parameter(d['bias'], (len(nodes),), ('bias',))
                state = m['inference_state']
                if state == {'type':'stateless'}:
                    # A stateless feedforward graph cannot contain an active cycle.
                    adjacency = {n:[] for n in nodes}
                    indegree = {n:0 for n in nodes}
                    for (a,b),enabled in zip(edges,m['edge_flags']):
                        if enabled:
                            adjacency[a].append(b)
                            indegree[b] += 1
                    queue = [n for n in nodes if indegree[n] == 0]
                    for n in queue:
                        for following in adjacency[n]:
                            indegree[following] -= 1
                            if indegree[following] == 0:
                                queue.append(following)
                    if len(queue) != len(nodes):
                        raise ValidationError('stateless graph contains active cycle')
                elif type(state) is dict and set(state) == {'type','values'} and state['type'] == 'recurrent':
                    parameter(state['values'],(len(nodes),),('inference_state',))
                else:
                    raise ValidationError('graph inference state descriptor incomplete/unsupported')
                return len(d['input_nodes']),len(d['output_nodes']),len(nodes)
            raise ValidationError(f'unsupported architecture type: {kind}')

        ni,no,layers = network(m['architecture'])
        supported_activations = {'identity','relu','tanh','sigmoid','elu','softplus'}
        if len(m['activations']) != layers or any(a not in supported_activations for a in m['activations']):
            raise ValidationError('explicit supported activation required for every layer/node')
        if self.kind == 'graph' and m['architecture']['type'] != 'graph':
            raise ValidationError('graph bundle requires graph architecture')
        if self.kind == 'vector_q' and no != len(m['action_grid']) * m['objective_dim']:
            raise ValidationError('vector-Q output shape must equal action count times objective_dim')
        normalizer = m['normalizer']
        if normalizer == {'type':'identity'}:
            pass
        elif set(normalizer) == {'type','mean','scale','epsilon'} and normalizer['type'] == 'standard':
            mean = parameter(normalizer['mean'],(ni,),('normalizer_mean',))
            scale = parameter(normalizer['scale'],(ni,),('normalizer_scale',))
            epsilon = normalizer['epsilon']
            if type(epsilon) not in (int,float) or epsilon < 0:
                raise ValidationError('normalizer epsilon must be finite and nonnegative')
            for array in (mean,scale):
                values = [v[0] for v in struct.iter_unpack('<'+_ARRAY_DTYPES[array.dtype][0],array.data)]
                if any(not math.isfinite(v) for v in values) or (array is scale and any(v + epsilon <= 0 for v in values)):
                    raise ValidationError('normalizer mean/scale must be finite with positive denominator')
        else:
            raise ValidationError('normalizer descriptor incomplete/unsupported; nonidentity arrays cannot be empty')
        decoder = m['decoder']
        if decoder == {'type':'identity'}:
            pass
        elif decoder.get('type') in ('clip','tanh_affine') and set(decoder) == {'type','low','high'}:
            low,high = decoder['low'],decoder['high']
            if type(low) is not list or type(high) is not list or len(low) != no or len(high) != no or any(type(a) not in (int,float) or type(b) not in (int,float) or a >= b for a,b in zip(low,high)):
                raise ValidationError('decoder requires finite ordered bounds matching output dimension')
        elif decoder.get('type') == 'affine' and set(decoder) == {'type','offset','scale'}:
            for key in ('offset','scale'):
                values = decoder[key]
                if type(values) is not list or len(values) != no or any(type(v) not in (int,float) for v in values):
                    raise ValidationError('affine decoder requires complete finite output vectors')
        else:
            raise ValidationError('decoder descriptor incomplete/unsupported')
        if used != set(self._arrays):
            raise ValidationError('unbound inference arrays not described by executable structure')

    def _descriptors(self, execution=False):
        descriptors, payloads, offset = [], [], 0
        metadata = self.metadata
        for name in sorted(self._arrays):
            array = self._arrays[name]
            dtype, raw = array.dtype, array.data
            if execution and dtype.startswith('f') and dtype != metadata['execution_dtype']:
                target = metadata['execution_dtype']
                try:
                    raw = b''.join(struct.pack('<' + _ARRAY_DTYPES[target][0], v[0])
                                   for v in struct.iter_unpack('<' + _ARRAY_DTYPES[dtype][0], raw))
                except (OverflowError, struct.error) as exc:
                    raise ValidationError('execution conversion overflows declared dtype') from exc
                dtype = target
                metadata['array_specs'][name]['dtype'] = target
            descriptors.append({'name': name, 'dtype': dtype, 'shape': list(array.shape), 'offset': offset,
                                'nbytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
            payloads.append(raw)
            offset += len(raw)
        return metadata, descriptors, payloads

    def _identity(self, execution=False):
        metadata, descriptors, _ = self._descriptors(execution)
        return hashlib.sha256(canonical_json({'kind': self.kind, 'metadata': metadata, 'arrays': descriptors})).hexdigest()

    @property
    def source_hash(self):
        return self._identity()

    @property
    def execution_hash(self):
        return self._identity(True)

    def encode(self, max_bytes=_MAX_BYTES):
        """Return canonical container bytes; cap includes every header/normalizer byte."""
        if type(max_bytes) is not int or not 0 < max_bytes <= _MAX_BYTES:
            raise ValidationError('bundle cap must be a positive integer within V3 maximum')
        metadata, descriptors, payloads = self._descriptors()
        header = canonical_json({'format': 'compact-storage-bundle-v3', 'kind': self.kind,
                                 'metadata': metadata, 'arrays': descriptors,
                                 'source_hash': self.source_hash, 'execution_hash': self.execution_hash})
        header_cap = _GRAPH_HEADER_CAP if self.kind == 'graph' else _HEADER_CAP
        if len(header) > header_cap or 12 + len(header) + sum(map(len,payloads)) > max_bytes:
            raise ValidationError('complete inference bundle exceeds byte/header cap')
        return _MAGIC + struct.pack('<I', len(header)) + header + b''.join(payloads)

    @classmethod
    def decode(cls, data, max_bytes=_MAX_BYTES):
        """Validate length, canonical header, array hashes and both identities before return."""
        if type(max_bytes) is not int or not 0 < max_bytes <= _MAX_BYTES:
            raise ValidationError('invalid bundle cap')
        if type(data) is not bytes or len(data) < 12 or len(data) > max_bytes or data[:8] != _MAGIC:
            raise ValidationError('invalid bundle prefix/length')
        size = struct.unpack_from('<I', data, 8)[0]
        # The largest permitted header bounds JSON parsing. Once kind is known,
        # enforce its narrower bound before allocating any decoded arrays.
        if size > _GRAPH_HEADER_CAP or 12 + size > len(data):
            raise ValidationError('invalid bounded bundle header')
        try:
            header = json.loads(data[12:12+size], object_pairs_hook=_unique_pairs,
                                parse_constant=lambda _: (_ for _ in ()).throw(ValidationError('nonfinite JSON')))
            canonical_json(header)
            if type(header) is not dict or set(header) != {'format','kind','metadata','arrays','source_hash','execution_hash'} or header['format'] != 'compact-storage-bundle-v3':
                raise ValidationError('unknown bundle schema')
            header_cap = _GRAPH_HEADER_CAP if header['kind'] == 'graph' else _HEADER_CAP
            if size > header_cap:
                raise ValidationError('invalid bounded bundle header')
            if type(header['arrays']) is not list:
                raise ValidationError('array descriptor list required')
            arrays, offset = {}, 0
            for d in header['arrays']:
                if type(d) is not dict or set(d) != {'name','dtype','shape','offset','nbytes','sha256'}:
                    raise ValidationError('invalid array descriptor fields')
                name = d['name']
                if type(name) is not str or name in arrays or type(d['shape']) is not list or type(d['offset']) is not int or type(d['nbytes']) is not int or d['offset'] != offset or d['nbytes'] < 0:
                    raise ValidationError('duplicate/invalid array descriptor')
                raw = data[12+size+offset:12+size+offset+d['nbytes']]
                if len(raw) != d['nbytes'] or hashlib.sha256(raw).hexdigest() != d['sha256']:
                    raise ValidationError('array content digest mismatch')
                arrays[name] = ArrayPayload(d['dtype'], tuple(d['shape']), raw)
                offset += d['nbytes']
            if 12 + size + offset != len(data):
                raise ValidationError('bundle contains unreferenced/truncated bytes')
            result = cls(header['kind'], arrays, header['metadata'])
            if result.source_hash != header['source_hash'] or result.execution_hash != header['execution_hash'] or result.encode(max_bytes) != data:
                raise ValidationError('bundle identity/canonical encoding mismatch')
            return result
        except (UnicodeError, ValueError, KeyError, TypeError, OverflowError, RecursionError) as exc:
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError('malformed bundle metadata') from exc
