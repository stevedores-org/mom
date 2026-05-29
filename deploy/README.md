# mom deploy/

Kustomize bases + overlays for running mom on Kubernetes. No Helm.

## Layout

```
deploy/
├── base/                    # Phase-1 minimal: Deployment + Service + SA + ConfigMap
└── overlays/
    └── orbstack/            # Local-dev (OrbStack k3s); add aks/eks/gke as needed
```

## Local dev — OrbStack

```bash
# 1. Build the OCI image via the flake. Must run on Linux (or via a Linux
#    remote builder); macOS hosts can't directly produce Linux layers.
nix build .#image

# 2. Load it into the local OrbStack containerd. `streamLayeredImage` writes
#    a streamable tarball; `result` is an executable that streams to stdout.
./result | docker load
# (OrbStack's docker shares the same containerd as its k3s, so the image is
#  immediately addressable by `mom:0.1.0` from inside the cluster.)

# 3. Apply the overlay.
kubectl --context orbstack apply -k deploy/overlays/orbstack

# 4. Verify.
kubectl --context orbstack -n mom-dev get pods
kubectl --context orbstack -n mom-dev port-forward svc/dev-mom 8080:80 &
curl http://localhost:8080/healthz
```

## Notes

- The base `image` field is a deliberate placeholder (`SET_BY_OVERLAY/...`); an
  un-overlaid apply fails fast.
- `MOM_DB_PATH=memory` in the ConfigMap → in-memory SurrealDB. Restart loses
  state. Persistent storage (PVC + path-style `MOM_DB_PATH`) is a Phase-2 add.
- Container is non-root (uid 65532, distroless), read-only rootfs, all caps
  dropped, `automountServiceAccountToken: false`. Standard Phase-1 hardening.
- Image build path is `flake.nix` → `dockerTools.streamLayeredImage` → packaged
  by `dockworker.ai` reading `dockworker.toml`. Do not introduce a Dockerfile.
