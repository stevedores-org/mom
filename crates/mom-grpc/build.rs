use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile protoc from source since it is not present in the system PATH
    std::env::set_var("PROTOC", protobuf_src::protoc());

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .file_descriptor_set_path(out_dir.join("memory_descriptor.bin"))
        .compile_protos(&["../../protos/memory.proto"], &["../../protos/"])?;
    Ok(())
}
