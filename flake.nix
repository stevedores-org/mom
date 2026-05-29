{
  description = "mom — Memory for All Autonomous Agents (workspace + OCI image)";

  nixConfig = {
    extra-substituters = [ "https://nix-cache.stevedores.org/" ];
    extra-trusted-public-keys = [
      "stevedores-1:ZEtb+wHYNR/LDmMDhF3/EpRZDNma8exY2b1TGZ6uS2A="
    ];
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    crane.url = "github:ipetkov/crane";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, crane, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rustfmt" "clippy" "rust-src" ];
        };

        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        src = craneLib.cleanCargoSource ./.;

        cargoToml = builtins.fromTOML (builtins.readFile ./Cargo.toml);
        pkgVersion = cargoToml.workspace.package.version;

        commonArgs = {
          inherit src;
          strictDeps = true;
          pname = "mom-workspace";
          version = pkgVersion;

          nativeBuildInputs = [ pkgs.pkg-config ];
          buildInputs = [ pkgs.openssl ]
            ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [ pkgs.libiconv ];
        };

        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        # mom-service produces a binary literally named `mom`.
        mom-service = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
          pname = "mom-service";
          cargoExtraArgs = "--bin mom";
        });

        workspaceClippy = craneLib.cargoClippy (commonArgs // {
          inherit cargoArtifacts;
          cargoClippyExtraArgs = "--workspace --all-targets -- -D warnings";
        });

        workspaceTests = craneLib.cargoNextest (commonArgs // {
          inherit cargoArtifacts;
          cargoNextestExtraArgs = "--workspace";
        });

        workspaceFmt = craneLib.cargoFmt { inherit src; };

        # OCI image — Linux-only. On macOS dev hosts, `nix build .#image` won't
        # work without a Linux remote builder; that's intentional. CI builds it
        # on a Linux runner and dockworker.ai ships the resulting tarball to the
        # registry.
        image = pkgs.dockerTools.streamLayeredImage {
          name = "mom";
          tag = pkgVersion;
          contents = [
            mom-service
            pkgs.cacert
          ];
          config = {
            Entrypoint = [ "${mom-service}/bin/mom" ];
            ExposedPorts = { "8080/tcp" = {}; };
            Env = [
              "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
              "RUST_LOG=info"
              "MOM_DB_PATH=memory"
            ];
            Labels = {
              "org.opencontainers.image.title" = "mom";
              "org.opencontainers.image.description" =
                "Memory for All Autonomous Agents — event-sourced memory kernel + retrieval engine";
              "org.opencontainers.image.source" = "https://github.com/stevedores-org/mom";
              "org.opencontainers.image.licenses" = "Apache-2.0";
              "stevedores.org/managed-by" = "dockworker";
            };
          };
        };

      in {
        checks = {
          inherit mom-service workspaceClippy workspaceTests workspaceFmt;
        };

        packages = {
          default = mom-service;
          inherit mom-service;
        } // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          inherit image;
          mom-image = image;
        };

        devShells.default = craneLib.devShell {
          checks = self.checks.${system};

          packages = with pkgs; [
            cargo-nextest
            cargo-watch
            # Kubernetes tooling for local-dev
            kubectl
            kustomize
            # Image-handling tooling for the OCI path
            skopeo
          ];
        };
      });
}
