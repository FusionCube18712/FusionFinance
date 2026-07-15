# AMD Compute Usage

FusionFinance used AMD compute for a technically meaningful recorded workload:
training an expanding walk-forward MLP ranker with PyTorch on ROCm/HIP. The
workload is a separate experiment and did not generate the displayed legacy
replay curves.

This page reports only fields supported by the receipt chain. The evidence identifies an AMD accelerator and its `gfx1100` architecture, but it does not identify an exact commercial product name.

## Receipt-reported workload

| Field | Receipt-backed value |
|---|---|
| Workload | Walk-forward quantitative model training |
| Model | Expanding walk-forward MLP ranker |
| Framework | PyTorch `2.9.1+gitff65f5b` |
| Runtime | ROCm/HIP `7.2.53211-e1a6bc5663` |
| Accelerator vendor | Advanced Micro Devices, Inc. `[AMD/ATI]` |
| GPU architecture | `gfx1100` |
| Visible device count | `1` |
| Reported VRAM | `51,522,830,336` bytes (approximately 48.0 GiB / 51.5 GB) |
| Walk-forward training runs | `72` |
| Recorded training time | `231.29` seconds |
| Accelerator utilization | Not captured in the receipt |
| Exact commercial SKU | Not proven; the raw device-name field is unavailable |

The ranker is the type of learned component that supplies a structured proposal
in the FusionFinance architecture. The preserved run demonstrates meaningful
AMD-backed training, but it is not lineage evidence for the checked-in legacy
curves.

The default judge experience is a reproducible static replay and does not require
a GPU, model endpoint, network connection, or API key. The receipts report the
recorded workload; they do not imply that every replay or hosted-dashboard
request executes on AMD hardware.

## Receipt chain

The public manifest is [`results/amd_compute.json`](../results/amd_compute.json).
The builder derives its vendor, architecture, runtime, memory, timing, and
retrain claims from the three source receipts, checks their cross-consistency,
and binds their exact bytes with SHA-256:

| Evidence | Source receipt | SHA-256 |
|---|---|---|
| Environment | [`evidence/amd/environment.json`](../evidence/amd/environment.json) | `cf7382dc842323579a87f031c09a9384db555c58f63a65dcf64bd23a36b10fcf` |
| Hardware | [`evidence/amd/hardware.json`](../evidence/amd/hardware.json) | `b1dab964b344b00e80780b19097652492a020b65539f8ee175f658a046db2ab8` |
| Training | [`evidence/amd/training.json`](../evidence/amd/training.json) | `4a9c246d0345b3a32f1660ad77a75809098f160f82d285de3d5a0e0969f85894` |

Verify the hashes, parsed receipt fields, and cross-receipt consistency from the
repository root:

```bash
python3 scripts/verify_amd.py
```

Rebuild the public replay, metric, decision, and AMD artifacts from the checked-in evidence with:

```bash
python3 scripts/build_fusion_artifacts.py
```

## Evidence boundary

- `rocm-smi` recorded AMD as the vendor, `gfx1100` as the architecture, and device ID `0x744b`, but its device-name lookup failed.
- FusionFinance therefore makes no claim that the device was a W7900, MI300, MI300X, or any other exact commercial SKU.
- The original probe filename contained `mi300`; that label was not hardware
  evidence. The public receipt is neutrally named `environment.json`, while its
  byte content and hash remain unchanged.
- No utilization percentage was captured, so the dashboard and submission materials must display utilization as unavailable rather than inventing a value.
- The receipt documents one completed training workload. It is not evidence of continuous AMD-backed live inference or production trading.
- SHA-256 demonstrates that the published bytes have not changed after the
  manifest was built. It is not a signed third-party attestation of the hardware
  or workload.

These limitations are deliberate. They make the AMD claim reproducibly
cross-checkable and keep the repository, deck, video, and demo consistent with
the available evidence.
