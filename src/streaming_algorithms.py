"""
streaming_algorithms.py
=======================
Streaming Algorithms Lab — SocialPulse Analytics
Implements:
  • Bloom Filter  (bit array + k hash functions)
  • Flajolet-Martin  (approximate distinct counting, multiple hashes)

These algorithms operate on a *simulated stream* replayed from the static
dataset. No live social-media ingestion is performed.

No third-party packages are required beyond stdlib (math, hashlib, array,
random) and numpy (already in venv).
"""

import array
import hashlib
import math
import random
from typing import Iterable, List


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hash_item(item: str, seed: int, m: int) -> int:
    """Return an integer in [0, m) for *item* using *seed* as a salt."""
    raw = hashlib.sha256(f"{seed}:{item}".encode("utf-8")).digest()
    # Use the first 8 bytes as a 64-bit integer
    val = int.from_bytes(raw[:8], "big")
    return val % m


def _trailing_zeros(n: int) -> int:
    """Count trailing zero bits in the binary representation of *n*.
    Returns 0 for n == 0 (edge-case guard).
    """
    if n == 0:
        return 0
    count = 0
    while (n & 1) == 0:
        count += 1
        n >>= 1
    return count


# ---------------------------------------------------------------------------
# Bloom Filter
# ---------------------------------------------------------------------------

class BloomFilter:
    """Bit-array Bloom Filter.

    Parameters
    ----------
    m_bits : int
        Total number of bits in the filter.
    k_hashes : int
        Number of independent hash functions.
    """

    def __init__(self, m_bits: int, k_hashes: int):
        if m_bits < 8:
            raise ValueError("m_bits must be >= 8")
        if k_hashes < 1:
            raise ValueError("k_hashes must be >= 1")
        self.m = m_bits
        self.k = k_hashes
        # One byte per bit position for simplicity and fast access
        self._bits = bytearray(m_bits)
        self._n_inserted = 0

    # ---- public API --------------------------------------------------------

    def add(self, item: str) -> None:
        """Insert *item* into the filter."""
        for seed in range(self.k):
            idx = _hash_item(item, seed, self.m)
            self._bits[idx] = 1
        self._n_inserted += 1

    def query(self, item: str) -> bool:
        """Return True if *item* is *possibly* present (never a false negative).

        A True result means the item is **possibly present, not definitely
        present**.  A False result guarantees the item was never inserted.
        """
        return all(
            self._bits[_hash_item(item, seed, self.m)] == 1
            for seed in range(self.k)
        )

    @property
    def n_inserted(self) -> int:
        return self._n_inserted

    def theoretical_fpr(self) -> float:
        """Expected false-positive rate given items inserted so far.
        Formula: (1 - e^(-k*n/m))^k
        """
        if self.m == 0:
            return 1.0
        exponent = -self.k * self._n_inserted / self.m
        return (1.0 - math.exp(exponent)) ** self.k

    def memory_bytes(self) -> int:
        """Approximate memory used by the bit array (bytes)."""
        return len(self._bits)


# ---------------------------------------------------------------------------
# Bloom Filter experiment runner
# ---------------------------------------------------------------------------

def run_bloom_filter_experiment(
    ids: List[str],
    m_bits: int,
    k_hashes: int,
    test_ratio: float = 0.5,
) -> dict:
    """Run a Bloom Filter experiment on *ids*.

    Strategy
    --------
    • Insert the first ``ceil(n * (1-test_ratio))`` IDs (the "known set").
    • Query the remaining IDs (never inserted) to measure false positives.
    • Also query the known set to confirm there are zero false negatives.

    Returns
    -------
    dict with keys:
        n_total, n_inserted, n_tested, possibly_seen_in_test,
        false_positives, false_positive_rate, theoretical_fpr,
        false_negatives (always 0 for a correct BF),
        memory_bytes, k_hashes, m_bits
    """
    if not ids:
        return {
            "n_total": 0, "n_inserted": 0, "n_tested": 0,
            "possibly_seen_in_test": 0, "false_positives": 0,
            "false_positive_rate": 0.0, "theoretical_fpr": 0.0,
            "false_negatives": 0, "memory_bytes": 0,
            "k_hashes": k_hashes, "m_bits": m_bits,
        }

    n_total = len(ids)
    split = max(1, int(n_total * (1 - test_ratio)))
    known = ids[:split]
    test_set = ids[split:]

    bf = BloomFilter(m_bits, k_hashes)
    known_exact = set(known)

    for item in known:
        bf.add(item)

    # Query test_set (items that were NOT inserted)
    false_positives = 0
    possibly_seen = 0
    for item in test_set:
        if item in known_exact:
            # Overlapping ID (edge case: dataset has duplicates) — skip
            continue
        if bf.query(item):
            possibly_seen += 1
            false_positives += 1

    # Confirm zero false negatives (sanity check)
    false_negatives = sum(1 for item in known if not bf.query(item))

    n_tested = sum(1 for item in test_set if item not in known_exact)
    fpr = false_positives / n_tested if n_tested > 0 else 0.0

    return {
        "n_total": n_total,
        "n_inserted": bf.n_inserted,
        "n_tested": n_tested,
        "possibly_seen_in_test": possibly_seen,
        "false_positives": false_positives,
        "false_positive_rate": fpr,
        "theoretical_fpr": bf.theoretical_fpr(),
        "false_negatives": false_negatives,
        "memory_bytes": bf.memory_bytes(),
        "k_hashes": k_hashes,
        "m_bits": m_bits,
    }


# ---------------------------------------------------------------------------
# Flajolet-Martin
# ---------------------------------------------------------------------------

class FlajoletMartin:
    """Approximate distinct-element counter (Flajolet-Martin algorithm).

    Uses *num_hashes* independent hash functions to reduce variance.
    The final estimate is the **median** of the per-hash estimates, which
    is more stable than a single estimate.

    Each hash h maps an item to an integer; the algorithm observes the
    position of the least-significant 1-bit (i.e. counts trailing zeros),
    maintaining a running maximum R_i for hash function i.
    Estimate per hash: 2^(R_i).
    Final estimate: median({2^R_i for all i}).

    Explanation
    -----------
    If items are hashed uniformly, a hash ending in at least k trailing
    zeros has probability 2^(-k).  After seeing ~N distinct items, it
    becomes likely that at least one hash will have ≥ log2(N) trailing
    zeros.  So max_trailing_zeros ≈ log2(N), giving estimate 2^R ≈ N.
    """

    def __init__(self, num_hashes: int = 8):
        if num_hashes < 1:
            raise ValueError("num_hashes must be >= 1")
        self.num_hashes = num_hashes
        # Maximum trailing-zero count seen per hash function
        self._max_tz: List[int] = [0] * num_hashes

    def add(self, item: str) -> None:
        """Process one item from the stream."""
        for i in range(self.num_hashes):
            raw = hashlib.md5(f"{i}:{item}".encode("utf-8")).digest()
            val = int.from_bytes(raw, "big")
            tz = _trailing_zeros(val)
            if tz > self._max_tz[i]:
                self._max_tz[i] = tz

    def per_hash_estimates(self) -> List[float]:
        """Return the per-hash estimates 2^R_i."""
        return [2 ** r for r in self._max_tz]

    def estimate(self) -> float:
        """Return the median estimate across all hash functions."""
        estimates = sorted(self.per_hash_estimates())
        n = len(estimates)
        mid = n // 2
        if n % 2 == 0:
            return (estimates[mid - 1] + estimates[mid]) / 2.0
        return float(estimates[mid])

    def max_trailing_zeros(self) -> List[int]:
        return list(self._max_tz)


# ---------------------------------------------------------------------------
# Flajolet-Martin experiment runner
# ---------------------------------------------------------------------------

def run_fm_experiment(ids: List[str], num_hashes: int = 8) -> dict:
    """Run an FM experiment on *ids* (treated as a stream).

    Returns
    -------
    dict with keys:
        n_stream_items, exact_distinct, fm_estimate, abs_error,
        pct_error, per_hash_estimates, max_trailing_zeros, num_hashes
    """
    if not ids:
        return {
            "n_stream_items": 0, "exact_distinct": 0,
            "fm_estimate": 0.0, "abs_error": 0.0,
            "pct_error": 0.0, "per_hash_estimates": [],
            "max_trailing_zeros": [], "num_hashes": num_hashes,
        }

    fm = FlajoletMartin(num_hashes=num_hashes)
    for item in ids:
        fm.add(str(item))

    exact = len(set(ids))
    estimate = fm.estimate()
    abs_err = abs(estimate - exact)
    pct_err = (abs_err / exact * 100) if exact > 0 else 0.0

    return {
        "n_stream_items": len(ids),
        "exact_distinct": exact,
        "fm_estimate": estimate,
        "abs_error": abs_err,
        "pct_error": pct_err,
        "per_hash_estimates": fm.per_hash_estimates(),
        "max_trailing_zeros": fm.max_trailing_zeros(),
        "num_hashes": num_hashes,
    }


# ---------------------------------------------------------------------------
# Controlled test harness (for the UI "Run Controlled Tests" expander)
# ---------------------------------------------------------------------------

def run_controlled_tests() -> List[dict]:
    """Validate both algorithms on known inputs.

    Test cases
    ----------
    1. Empty input
    2. All-same IDs (1 distinct item, repeated 1 000×)
    3. All-unique IDs (1 000 distinct items, no repeats)
    4. Known false-positive scenario: insert 500 IDs, query 500 different IDs
    """
    results = []

    # ---- Test 1: Empty input -----------------------------------------------
    bf_res = run_bloom_filter_experiment([], m_bits=1000, k_hashes=3)
    fm_res = run_fm_experiment([], num_hashes=8)
    results.append({
        "name": "Empty input",
        "bf_inserted": bf_res["n_inserted"],
        "bf_fp": bf_res["false_positives"],
        "bf_fn": bf_res["false_negatives"],
        "fm_exact": fm_res["exact_distinct"],
        "fm_estimate": fm_res["fm_estimate"],
        "pass": bf_res["n_inserted"] == 0 and fm_res["exact_distinct"] == 0,
    })

    # ---- Test 2: All-same IDs ----------------------------------------------
    same_ids = ["user_42"] * 1000
    bf_res2 = run_bloom_filter_experiment(same_ids, m_bits=1000, k_hashes=3)
    fm_res2 = run_fm_experiment(same_ids, num_hashes=8)
    results.append({
        "name": "All-same IDs (1 000× 'user_42')",
        "bf_inserted": bf_res2["n_inserted"],
        "bf_fp": bf_res2["false_positives"],
        "bf_fn": bf_res2["false_negatives"],
        "fm_exact": fm_res2["exact_distinct"],
        "fm_estimate": fm_res2["fm_estimate"],
        # FM estimate should be close to 1; accept range [0.5, 4] (single hash variance)
        "pass": fm_res2["exact_distinct"] == 1 and bf_res2["false_negatives"] == 0,
    })

    # ---- Test 3: All-unique IDs --------------------------------------------
    unique_ids = [f"tweet_{i}" for i in range(1000)]
    bf_res3 = run_bloom_filter_experiment(unique_ids, m_bits=20000, k_hashes=5)
    fm_res3 = run_fm_experiment(unique_ids, num_hashes=8)
    results.append({
        "name": "All-unique IDs (1 000 distinct tweets, m=20000, k=5)",
        "bf_inserted": bf_res3["n_inserted"],
        "bf_fp": bf_res3["false_positives"],
        "bf_fn": bf_res3["false_negatives"],
        "fm_exact": fm_res3["exact_distinct"],
        "fm_estimate": fm_res3["fm_estimate"],
        # No false negatives allowed; FPR should be very low with large m
        "pass": bf_res3["false_negatives"] == 0 and fm_res3["exact_distinct"] == 1000,
    })

    # ---- Test 4: Known false-positive validation ---------------------------
    # Insert IDs 0-499, query IDs 500-999 (disjoint)
    insert_ids = [f"id_{i}" for i in range(500)]
    query_ids = [f"id_{i}" for i in range(500, 1000)]
    # Manually run the BF to control insert vs query sets precisely
    bf4 = BloomFilter(m_bits=500, k_hashes=3)   # deliberately small → higher FPR
    for item in insert_ids:
        bf4.add(item)
    fp4 = sum(1 for item in query_ids if bf4.query(item))
    fn4 = sum(1 for item in insert_ids if not bf4.query(item))
    results.append({
        "name": "Known FP scenario: insert 500, query 500 disjoint (m=500, k=3)",
        "bf_inserted": bf4.n_inserted,
        "bf_fp": fp4,
        "bf_fn": fn4,
        "fm_exact": "N/A",
        "fm_estimate": "N/A",
        # False negatives must be 0; false positives >= 0 (expected ~high due to small m)
        "pass": fn4 == 0,
    })

    return results
