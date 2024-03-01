// SHA-256 for 32-bit aligned messages

fn swap_endianess32(val: u32) -> u32 {
    return ((val>>24u) & 0xffu) | ((val>>8u) & 0xff00u) | ((val<<8u) & 0xff0000u) | ((val<<24u) & 0xff000000u);
}

fn shw(x: u32, n: u32) -> u32 {
    return (x << (n & 31u)) & 0xffffffffu;
}

fn r(x: u32, n: u32) -> u32 {
    return (x >> n) | shw(x, 32u - n);
}

fn g0(x: u32) -> u32 {
    return r(x, 7u) ^ r(x, 18u) ^ (x >> 3u);
}

fn g1(x: u32) -> u32 {
    return r(x, 17u) ^ r(x, 19u) ^ (x >> 10u);
}

fn s0(x: u32) -> u32 {
    return r(x, 2u) ^ r(x, 13u) ^ r(x, 22u);
}

fn s1(x: u32) -> u32 {
    return r(x, 6u) ^ r(x, 11u) ^ r(x, 25u);
}

fn maj(a: u32, b: u32, c: u32) -> u32 {
    return (a & b) ^ (a & c) ^ (b & c);
}

fn ch(e: u32, f: u32, g: u32) -> u32 {
    return (e & f) ^ ((~e) & g);
}

@group(0)
@binding(0)
var<storage, read_write> messages: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> hashes: array<u32>;

@group(0)
@binding(2)
var<storage, read> num_messages: u32;

@group(0)
@binding(3)
var<storage, read> message_sizes: array<u32>;

@group(0)
@binding(4)
var<storage, read> k: array<u32>;

@compute
@workgroup_size(32)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= num_messages) {
        return;
    }
    let message_base_index = index * message_sizes[1];
    let hash_base_index = index * (256u / 32u);

    // padding
    messages[message_base_index + message_sizes[0]] = 0x00000080u;
    for (var i = message_sizes[0] + 1; i < message_sizes[1] - 2; i++){
        messages[message_base_index + i] = 0x00000000u;
    }
    messages[message_base_index + message_sizes[1] - 2] = 0u;
    messages[message_base_index + message_sizes[1] - 1] = swap_endianess32(message_sizes[0] * 32u);
    
    // processing
    hashes[hash_base_index] = 0x6a09e667u;
    hashes[hash_base_index + 1] = 0xbb67ae85u;
    hashes[hash_base_index + 2] = 0x3c6ef372u;
    hashes[hash_base_index + 3] = 0xa54ff53au;
    hashes[hash_base_index + 4] = 0x510e527fu;
    hashes[hash_base_index + 5] = 0x9b05688cu;
    hashes[hash_base_index + 6] = 0x1f83d9abu;
    hashes[hash_base_index + 7] = 0x5be0cd19u;

    let num_chunks = (message_sizes[1] * 32u) / 512u;
    for (var i = 0u; i < num_chunks; i++) {
        let chunk_index = i * (512u/32u);
        var w = array<u32,64u>();
        for (var j = 0u; j < 16u; j++) {
            w[j] = swap_endianess32(messages[message_base_index + chunk_index + j]);
        }
        for (var j = 16u; j < 64u; j++){
            w[j] = w[j - 16u] + g0(w[j - 15u]) + w[j - 7u] + g1(w[j - 2u]);
        }
        var a = hashes[hash_base_index];
        var b = hashes[hash_base_index + 1];
        var c = hashes[hash_base_index + 2];
        var d = hashes[hash_base_index + 3];
        var e = hashes[hash_base_index + 4];
        var f = hashes[hash_base_index + 5];
        var g = hashes[hash_base_index + 6];
        var h = hashes[hash_base_index + 7];
        for (var j = 0u; j < 64u; j++){
            let t2 = s0(a) + maj(a, b, c);
            let t1 = h + s1(e) + ch(e, f, g) + k[j] + w[j];
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        hashes[hash_base_index] += a;
        hashes[hash_base_index + 1] += b;
        hashes[hash_base_index + 2] += c;
        hashes[hash_base_index + 3] += d;
        hashes[hash_base_index + 4] += e;
        hashes[hash_base_index + 5] += f;
        hashes[hash_base_index + 6] += g;
        hashes[hash_base_index + 7] += h;
    }
    hashes[hash_base_index] = swap_endianess32(hashes[hash_base_index]);
    hashes[hash_base_index + 1] = swap_endianess32(hashes[hash_base_index + 1]);
    hashes[hash_base_index + 2] = swap_endianess32(hashes[hash_base_index + 2]);
    hashes[hash_base_index + 3] = swap_endianess32(hashes[hash_base_index + 3]);
    hashes[hash_base_index + 4] = swap_endianess32(hashes[hash_base_index + 4]);
    hashes[hash_base_index + 5] = swap_endianess32(hashes[hash_base_index + 5]);
    hashes[hash_base_index + 6] = swap_endianess32(hashes[hash_base_index + 6]);
    hashes[hash_base_index + 7] = swap_endianess32(hashes[hash_base_index + 7]);    
}
