# Quality, Tool-Use & RAM Compression Gate: `Fractal-64-BF16-14.35GiB` vs `LUT-VQ3-64-BF16-13.08GiB`

- **Release Gate Verdict:** **❌ HOLD (Requires Tuning Before Release)**
- **Quality & Tool-Use Retention:** `100.0%` (`80.0/80` vs `80.0/80`)
- **Decode Speed Ratio:** `96.6%` (`24.18 tok/s` vs `25.02 tok/s`)
- **RAM Footprint Delta:** `-1.23 GiB` (`16.04 GiB` vs `17.27 GiB`)

| Metric / Suite | `Fractal-64-BF16-14.35GiB` | `LUT-VQ3-64-BF16-13.08GiB` | Delta |
| :--- | :---: | :---: | :---: |
| **1. Agentic Tool-Calling (`30` pts)** | `30.0` | **`30.0`** | `+0.0` |
| **2. Executable Code Gen (`30` pts)** | `30.0` | **`30.0`** | `+0.0` |
| **3. Long-Context Needle & Math (`20` pts)** | `20.0` | **`20.0`** | `+0.0` |
| **Quality Subtotal (`80` pts)** | `80.0` | **`80.0`** | `100.0% parity` |
| **Decode Throughput (`tok/s`)** | `25.02 tok/s` | **`24.18 tok/s`** | `-3.4%` |
| **Stockfish-PVS Acceptance (`%`)** | `59.09%` | **`70.19%`** | `+11.1%` |
| **Resident RAM (`VmRSS GiB`)** | `17.27 GiB` | **`16.04 GiB`** | **`-1.23 GiB`** |
