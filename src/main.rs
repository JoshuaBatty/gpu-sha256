use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use sha2::{Sha256, Digest};
use std::borrow::Cow;
use wgpu::util::DeviceExt;
    
const NUM_HASHES: usize = 2_000_000;

// Constants for SHA-256
// These are read in the compute shader
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error>> { 
    // Test data
    let mut messages = vec![];
    for i in 0..NUM_HASHES {
        // vary the bytes based on the loop index
        let byte0 = (i & 0xff) as u8;
        let byte1 = ((i >> 8) & 0xff) as u8;
        let byte2 = ((i >> 16) & 0xff) as u8;
        let byte3 = ((i >> 24) & 0xff) as u8;
        messages.push(vec![byte0, byte1, byte2, byte3]);
    }
    let messages_to_hash = messages.len() as u32;

    // Hash with rust sha256
    let mut rust_hashes = vec![];
    let now = std::time::Instant::now();
    for message in &messages {
        let mut hasher = Sha256::new();
        hasher.update(message);
        let result = hasher.finalize();
        rust_hashes.push(format!("{:x}",result));
    }
    println!("Rust sequential sha256 took: {:?}", now.elapsed());

    // do the above but with rayon
    let now = std::time::Instant::now();
    let _: Vec<_> = messages.par_iter()
        .map(|message| {
            let mut hasher = Sha256::new();
            hasher.update(message);
            let result = hasher.finalize();
            format!("{:x}",result)
        })
        .collect();
    println!("Rust parallel sha256 took: {:?}", now.elapsed());

    let message_sizes = get_message_sizes(&messages[0]);

    // Add padding to each message and convert directly to u32
    let messages_u32: Vec<u32> = messages.iter()
        .flat_map(|message| pad_message_for_sha256(message)) // Pad each message
        .collect::<Vec<u8>>() // Collect padded messages into a single Vec<u8>
        .chunks_exact(4) // Split the Vec<u8> into 4-byte chunks
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap())) // Convert each chunk into u32
        .collect(); // Collect the u32 values into a Vec<u32>
    
    let result = execute_gpu(&messages_u32, message_sizes, messages_to_hash).await.unwrap();

    // Convert the entire result into bytes first
    let result_bytes = bytemuck::cast_slice::<u32, u8>(&result);

    // Then convert each byte to a hexadecimal string
    let mut hash_hex_strings: Vec<String> = Vec::new();
    for chunk in result_bytes.chunks(32) { // Assuming 32 bytes per hash
        let hash_hex_str = chunk.iter()
            .map(|byte| format!("{:02x}", byte))
            .collect::<String>();
        hash_hex_strings.push(hash_hex_str);
    }

    // eprintln!("Rust sha256 hashes: {:#?}", rust_hashes);
    // eprintln!("GPU sha256 hashes: {:#?}", hash_hex_strings);

    assert!(rust_hashes == hash_hex_strings);
    Ok(())
}

#[cfg_attr(test, allow(dead_code))]
async fn execute_gpu(
    messages: &[u32],
    message_sizes: Vec<u32>,
    messages_to_hash: u32,
) -> Option<Vec<u32>> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    drop(instance);
    drop(adapter);
    execute_gpu_inner(&device, &queue, messages, message_sizes, messages_to_hash).await
}

async fn execute_gpu_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    messages: &[u32],
    message_sizes: Vec<u32>,
    messages_to_hash: u32,
) -> Option<Vec<u32>> {
    let timestamp_period: f32 = queue.get_timestamp_period();

    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    let result_buffer_size = (256 / 8) * messages_to_hash as u64; // 32 bytes per hash
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: result_buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let messages_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Messages Buffer"),
        contents: bytemuck::cast_slice(messages),
        usage: wgpu::BufferUsages::STORAGE
    });
    
    // Num messages buffer
    let num_messages_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Num Messages Buffer"),
        contents: bytemuck::cast_slice(&[messages.len() as u32]),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let message_sizes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Message Sizes Buffer"),
        contents: bytemuck::cast_slice(&message_sizes),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let hashes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Result Buffer"),
        size: result_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let k_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("K Buffer"),
        contents: bytemuck::cast_slice(&K), 
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, 
    });
    
    let timestamp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Timestamps buffer"),
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let timestamp_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    timestamp_readback_buffer.unmap();

    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
    });

    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: messages_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: hashes_buffer.as_entire_binding(),
            },  
            wgpu::BindGroupEntry {
                binding: 2,
                resource: num_messages_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: message_sizes_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: k_buffer.as_entire_binding(),
            },    
        ],
    });

    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        count: 2,
        ty: wgpu::QueryType::Timestamp,
    });

    //----------
    let threads_per_group = 32; // This should match the compute shader's @workgroup_size
    let num_workgroups_x = (messages_to_hash + threads_per_group - 1) / threads_per_group;

    println!("Total messages to hash: {}", messages_to_hash);
    println!("Dispatching {} workgroups with {} threads each", num_workgroups_x, threads_per_group);

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("compute sha256 hashes");
        cpass.write_timestamp(&queries, 0);
        cpass.dispatch_workgroups(num_workgroups_x, 1, 1);
        cpass.write_timestamp(&queries, 1);
    }
    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    encoder.copy_buffer_to_buffer(&hashes_buffer, 0, &staging_buffer, 0, hashes_buffer.size());
    encoder.resolve_query_set(&queries, 0..2, &timestamp_buffer, 0);
    encoder.copy_buffer_to_buffer(&timestamp_buffer,0,&timestamp_readback_buffer,0,timestamp_buffer.size());

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = staging_buffer.slice(..);
    let timestamp_slice = timestamp_readback_buffer.slice(..);

    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    timestamp_slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    // Awaits until `buffer_future` can be read from
    if let Ok(Ok(())) = receiver.recv_async().await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        let timing_data = timestamp_slice.get_mapped_range();

        // Since contents are got in bytes, this converts these bytes back to u32
        let result = bytemuck::cast_slice(&data).to_vec();

        let timings = timing_data
            .chunks_exact(8)
            .map(|b| u64::from_ne_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        
        // Unmap buffers from memory and drop them
        drop(data);
        drop(timing_data);
        staging_buffer.unmap(); 
        timestamp_readback_buffer.unmap();

        println!(
            "GPU Compute Shader: {:?}",
            std::time::Duration::from_nanos(
                ((timings[1] - timings[0]) as f64 * f64::from(timestamp_period)) as u64
            )
        );

        // Returns data from buffer
        Some(result)
    } else {
        panic!("failed to run compute on gpu!")
    }
}


fn pad_message_for_sha256(message: &[u8]) -> Vec<u8> {
    let mut padded_message = message.to_vec();
    // Pad the message according to SHA-256 specifications
    // Append the bit '1' to the message
    padded_message.push(0x80);
    // Append '0' bits until the message length in bits is 448 mod 512
    while (padded_message.len() * 8) % 512 != 448 {
        padded_message.push(0x00);
    }
    // Append the message length as a 64-bit big-endian integer, making the total length a multiple of 512 bits
    let bit_len = (message.len() as u64) * 8;
    padded_message.extend_from_slice(&bit_len.to_be_bytes());
    padded_message
}

fn get_message_sizes(bytes: &[u8]) -> Vec<u32> {
    let len_bit = bytes.len() as u64 * 8;
    let k = 512 - (len_bit + 1 + 64) % 512;
    let padding = 1 + k + 64;
    let len_bit_padded = len_bit + padding;
    vec![len_bit as u32 / 32, len_bit_padded as u32 / 32]
}