# Cache Eviction Policy Analysis: LRU vs LFU vs ML

A comparison of traditional cache eviction policies (LRU, LFU) against a
learned eviction policy (Random Forest regression trained on oracle
look-ahead features) on a synthetic Zipfian access trace.

**TL;DR:** LFU marginally outperforms a 200-tree Random Forest on this
workload (0.56 vs 0.56 hit rate at cache size 10), while being roughly
**14,000x faster per eviction decision**. The interesting result isn't
"ML wins" — it's *when* the added complexity is worth it, and when it
clearly isn't.

## Results

**Table 1.** Hit rate and average per-eviction latency at cache size 10.

<table border="1" cellspacing="0" cellpadding="6">
  <tr>
    <th>Algorithm</th>
    <th>Hit Rate (cache=10)</th>
    <th>Avg Latency / Eviction</th>
  </tr>
  <tr>
    <td>LRU</td>
    <td>0.301</td>
    <td>0.00037s</td>
  </tr>
  <tr>
    <td>LFU</td>
    <td>0.561</td>
    <td>0.0000004s</td>
  </tr>
  <tr>
    <td>ML Cache</td>
    <td>0.559</td>
    <td>0.00512s</td>
  </tr>
</table>

**Table 2.** Hit rate by cache size for each eviction policy.

<table border="1" cellspacing="0" cellpadding="6">
  <tr>
    <th>Cache Size</th>
    <th>LRU</th>
    <th>LFU</th>
    <th>ML</th>
  </tr>
  <tr>
    <td>5</td>
    <td>0.170</td>
    <td>0.436</td>
    <td>0.472</td>
  </tr>
  <tr>
    <td>10</td>
    <td>0.301</td>
    <td>0.561</td>
    <td>0.559</td>
  </tr>
  <tr>
    <td>20</td>
    <td>0.400</td>
    <td>0.623</td>
    <td>0.616</td>
  </tr>
  <tr>
    <td>50</td>
    <td>0.511</td>
    <td>0.670</td>
    <td>0.680</td>
  </tr>
  <tr>
    <td>100</td>
    <td>0.571</td>
    <td>0.696</td>
    <td>0.699</td>
  </tr>
</table>

![Hit rate vs cache size](images/hit_rate_vs_cache_size.png)
*Figure 1. Hit rate vs. cache size for LRU, LFU, and ML eviction on the same Zipfian trace.*
> LRU trails both other policies at every size — recency alone is a weak
> signal on a Zipf-distributed trace. LFU and ML are nearly tied
> throughout, and the gap between them narrows further as cache size
> grows: at size 5, ML holds a real edge (0.472 vs 0.436); by size 100,
> LFU and ML are effectively tied (0.696 vs 0.699). The takeaway: ML's
> advantage, such as it is, matters most when the cache is small
> relative to the working set — exactly where every eviction decision
> counts most.

## Why this is interesting

Most "ML cache" writeups stop at "the model beats the baseline." This one
doesn't, on purpose:

- **LRU is a weak baseline here.** On a Zipf-distributed trace, a small
  number of items account for most traffic, so pure recency is a bad
  eviction signal. LFU captures this directly; LRU doesn't.
- **The Random Forest is trained on oracle features** (forward-looking
  reuse distance, computed only from the training portion of the trace)
  and predicts which cached item is least likely to be needed soon —
  effectively an offline-trained approximation of Belady's algorithm.
- **A synthetic scan injection** (a burst of sequential unique IDs) is
  added to the trace specifically to probe scan resistance, a known
  weakness of history-based ML eviction.
- **The latency cost is measured, not assumed.** Calling
  `model.predict()` on every eviction is ~14,000x slower than an LFU
  frequency-count lookup. In a real cache, that overhead has to be
  weighed against the hit-rate gain, and here it clearly isn't worth it.

## Where ML eviction would actually help

Frequency-based policies like LFU are strong exactly when historical
frequency stays a reliable predictor of future access — i.e., stable,
Zipf-like workloads. ML-based eviction becomes more competitive on
**non-stationary** workloads where popularity shifts over time and
recency/reuse-distance features carry more signal than raw frequency
counts. That's a natural extension of this analysis, not something this
trace tests.

## Methodology

1. Generate a synthetic Zipfian access trace (`~10k` requests) with an
   injected sequential scan to test scan resistance.
2. Split the trace 80/20 into `train_trace` / `test_trace` **before** any
   feature engineering, so the ML model never sees the evaluation region
   during training (avoids train/test leakage).
3. Build oracle features (recency, frequency, reuse distance) from
   `train_trace` only, with the label being the true forward distance to
   next reuse.
4. Train a `RandomForestRegressor` (200 trees) to predict reuse distance;
   at eviction time, evict the cached item with the highest predicted
   reuse distance.
5. Run LRU, LFU, and the ML policy over the same warm-up/eval trace split
   and cache-size sweep, measuring hit rate and per-eviction latency for
   each.

## Reproducing this

```bash
pip install -r requirements.txt
jupyter notebook cache_eviction_analysis.ipynb
```

The notebook runs cleanly top-to-bottom with a fixed random seed, so
results are reproducible across runs. The `cache_sizes` sweep
(`hit_rate_vs_cache_size.png`) uses precomputed results by default since
it involves thousands of `model.predict()` calls (~5 minutes to
regenerate); the full loop is left in the notebook, commented, if you
want to re-run it from scratch.

## Limitations

- The Random Forest is trained only on items that recur at least once
  (items that never come back are dropped from training) — this is a
  mild survivorship bias worth noting if extending this to real traces.
- All results are from a single synthetic Zipfian trace; conclusions
  about "when ML eviction helps" are hypotheses this setup is designed
  to illustrate, not something validated on real production traces.
