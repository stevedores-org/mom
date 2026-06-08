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
# 1. Image must already be in GAR. Push manually from a Linux host:
#      gcloud auth configure-docker us-docker.pkg.dev
#      nix build .#image
#      ./result | skopeo copy docker-archive:/dev/stdin \
#        docker://us-docker.pkg.dev/gcp-lornu-ai/stevedores/mom:$(yq -p toml -o json Cargo.toml | jq -r '.workspace.package.version')

# 2. Authenticate to GCP if your token has expired.
gcloud auth login
gcloud container clusters get-credentials lornu-gke-prod \
  --region us-central1 --project gcp-lornu-ai

# 3. Apply the overlay. (kubectl apply -k works directly here as there are no out-of-tree references)
kubectl --context gke_gcp-lornu-ai_us-central1_lornu-gke-prod apply -k deploy/overlays/gke

# 4. Verify.
kubectl --context gke_gcp-lornu-ai_us-central1_lornu-gke-prod -n mom \
  get deploy,svc,sa,cm,pods,networkpolicy,pdb
kubectl --context gke_gcp-lornu-ai_us-central1_lornu-gke-prod -n mom \
  port-forward svc/mom 8080:80 &
curl http://localhost:8080/healthz
```

### Production Digest Pinning (L2 & L7)
To align with production hardening guidelines, do not rely on mutable version tags. Pin the image to a specific SHA256 digest in the overlay:
```bash
(cd deploy/overlays/gke && kustomize edit set image SET_BY_OVERLAY/mom=us-docker.pkg.dev/gcp-lornu-ai/stevedores/mom@sha256:<sha256-hash>)
```
Re-apply the overlay once the image target is pinned by digest.

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

## Troubleshooting (L5)

### Image Pull Failures (`ErrImagePull` or `ImagePullBackOff`)
This is usually caused by incorrect Workload Identity configurations or missing IAM roles on the Google Service Account (GSA).
1. Verify the GKE cluster has Workload Identity enabled:
   ```bash
   gcloud container clusters describe lornu-gke-prod --region us-central1 --format="value(workloadIdentityConfig.workloadPool)"
   ```
2. Verify the KSA is annotated correctly:
   ```bash
   kubectl -n mom get sa mom -o yaml
   # Look for: iam.gke.io/gcp-service-account: mom@gcp-lornu-ai.iam.gserviceaccount.com
   ```
3. Check the IAM policy binding on the GSA:
   ```bash
   gcloud iam service-accounts get-iam-policy mom@gcp-lornu-ai.iam.gserviceaccount.com
   # Ensure gcp-lornu-ai.svc.id.goog[mom/mom] has roles/iam.workloadIdentityUser
   ```
4. Check Artifact Registry reader role permissions:
   ```bash
   gcloud artifacts repositories get-iam-policy stevedores --location us --project gcp-lornu-ai
   # Ensure the GSA has roles/artifactregistry.reader
   ```

### Pod Remaining in `Pending` State
The `mom` namespace enforces the **restricted** Pod Security Standard (PSS). If the pod stays pending, check for a PSS rejection event:
```bash
kubectl -n mom describe pod -l app.kubernetes.io/name=mom
```
Ensure that no container settings bypass the security context:
- `runAsNonRoot: true`
- `allowPrivilegeEscalation: false`
- `readOnlyRootFilesystem: true`
- `capabilities.drop: ["ALL"]`

## Notes & Design Constraints

- **Nix Caching Warning**: If you encounter warnings about untrusted Nix substituters when running local builds, follow the onboarding instructions in the [Nix Trusted Users Guide](https://github.com/lornu-ai/container-track/blob/main/docs/NIX_TRUSTED_USERS.md).
- **State Loss Warning (L8)**: Because the `mom` service operates in-memory (`MOM_DB_PATH=memory`), any pod restart will result in total loss of state. Do not scale replicas beyond 1. Persistent storage support (PVC + file-backed SurrealDB) is scheduled for Phase 2.
- **Eviction Protection**: A `PodDisruptionBudget` is defined to prevent voluntary node drains from evicting the single `mom` replica automatically, avoiding state loss during maintenance.
- **Network Isolation**: Ingress is restricted via `NetworkPolicy` to only allow traffic from within the namespace and from authorized agent workloads (`ciso-agent`, `bookkeeping`, `agent-scheduler`).
- Container runs as non-root (uid 65532, distroless), read-only rootfs, all capabilities dropped.
- Image build path is `flake.nix` → `dockerTools.streamLayeredImage` → packaged by `dockworker.ai` reading `dockworker.toml`. **Do not introduce a Dockerfile.**

