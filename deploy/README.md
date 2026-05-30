# mom deploy/

Kustomize bases + overlays for running mom on Kubernetes. No Helm.

## Layout

```
deploy/
├── base/                    # Phase-1 minimal: Deployment + Service + SA + ConfigMap
└── overlays/
    └── gke/                 # gke_gcp-lornu-ai_us-central1_lornu-gke-prod
```

## GKE — production cluster

```bash
# 1. Image must already be in GAR (build path via dockworker.ai + flake.nix).
#    For ad-hoc local push from a Linux host:
#      nix build .#image
#      ./result | skopeo copy docker-archive:/dev/stdin \
#        docker://us-docker.pkg.dev/gcp-lornu-ai/stevedores/mom:$(yq -p toml -o json Cargo.toml | jq -r '.workspace.package.version')
#    For CI: dockworker.ai does this automatically per the dockworker.toml target.

# 2. Authenticate to GCP if your token has expired.
gcloud auth login
gcloud container clusters get-credentials lornu-gke-prod \
  --region us-central1 --project gcp-lornu-ai

# 3. Apply the overlay.
kustomize build deploy/overlays/gke \
  | kubectl --context gke_gcp-lornu-ai_us-central1_lornu-gke-prod apply -f -

# 4. Verify.
kubectl --context gke_gcp-lornu-ai_us-central1_lornu-gke-prod -n mom \
  get deploy,svc,sa,cm,pods
kubectl --context gke_gcp-lornu-ai_us-central1_lornu-gke-prod -n mom \
  port-forward svc/mom 8080:80 &
curl http://localhost:8080/healthz
```

## Platform prerequisites

Before the overlay applies cleanly, the platform team must satisfy:

1. **GSA `mom@gcp-lornu-ai.iam.gserviceaccount.com`** with `roles/artifactregistry.reader`
   on `projects/gcp-lornu-ai/locations/us/repositories/stevedores`.
2. **KSA→GSA Workload Identity binding**:
   ```bash
   gcloud iam service-accounts add-iam-policy-binding \
     mom@gcp-lornu-ai.iam.gserviceaccount.com \
     --role roles/iam.workloadIdentityUser \
     --member "serviceAccount:gcp-lornu-ai.svc.id.goog[mom/mom]"
   ```

## Notes

- `MOM_DB_PATH=memory` in the ConfigMap is currently decorative — on `main` the
  store ignores the path and always uses in-memory SurrealDB. See #17.
- The image build / OCI entrypoint convention is reflected in `flake.nix`; see
  #16 for the divergence with `dockworker.toml` to be resolved.
- Container is non-root (uid 65532, distroless), read-only rootfs, all caps
  dropped. The Namespace has Pod Security Standards enforcement set to
  `restricted`.
- Image build path is `flake.nix` → `dockerTools.streamLayeredImage` → packaged
  by `dockworker.ai` reading `dockworker.toml`. **Do not introduce a Dockerfile.**
- Persistent storage (PVC + path-style `MOM_DB_PATH`) is a Phase-2 add — gated
  on #17 being resolved.
