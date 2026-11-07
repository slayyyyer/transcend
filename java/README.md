# Java runtime

Offline English↔French translation through ONNX Runtime. No Python, no network,
no model download at runtime.

## Requirements

* Java 17 or newer (the code targets release 17).
* Windows x64, Linux x64 or macOS (ONNX Runtime ships the native libraries in
  the Maven artifact).
* About 3× the ONNX file size of free RAM for loading; the 9.4M-parameter model
  package is roughly 40 MB.

## Build and test

```bash
cd java
mvn -B package                     # compiles, runs unit tests, copies deps to target/lib
mvn -B test                        # unit tests only
mvn -B test -Dtranscend.model.dir=../models/accel_export   # + integration parity test
```

If Maven is not on `PATH`, configure your JDK 17+ and Maven installation.

## CLI

```bash
CP="target/classes:target/lib/*"  # use the classpath separator for your platform
java -cp "$CP" org.transcend.nmt.Main translate \
  --model ../models/accel_export --from en --to fr \
  "The train leaves at eight in the morning."

java -cp "$CP" org.transcend.nmt.Main batch \
  --model ../models/accel_export --from en --to fr --file inputs.txt

java -cp "$CP" org.transcend.nmt.Main bench \
  --model ../models/accel_export --from en --to fr --count 20 --out report.json
```

## API

```java
try (Translator translator = Translator.open(
        Path.of("models/accel_export"),
        RuntimeOptions.builder().intraOpThreads(4).maxConcurrency(4).build())) {

    TranslationResult result = translator.translate(
            "She works in a small bookshop near the river.",
            Language.EN, Language.FR, TranslationOptions.defaults());

    System.out.println(result.text());
    // result.modelId(), result.terminationReason(), result.sourceTokens(),
    // result.targetTokens(), result.warnings()
}
```

Batch translation preserves input order:

```java
List<TranslationResult> results =
        translator.translateBatch(texts, Language.FR, Language.EN, TranslationOptions.defaults());
```

### Behaviour

* Only EN→FR and FR→EN are accepted; anything else throws
  `IllegalArgumentException`.
* `null` text throws. Empty or whitespace-only input returns an empty string with
  termination reason `empty_input` and no model call.
* Input that exceeds the packaged source-token limit throws
  `InputTooLongException`; it is never silently truncated.
* Termination reason `max_length` means the decoder hit the output limit. A
  warning is added and `completedNaturally()` returns false.
* `TranslationOptions` can set `maxNewTokens`, `beamSize` and `lengthPenalty`;
  unset values come from the packaged `generation.json`.
* `Translator` is thread-safe; concurrent calls are bounded by
  `RuntimeOptions.maxConcurrency` (a fair semaphore). Batch calls run their items
  sequentially inside the calling thread.
* The package is verified against `manifest.json` SHA-256 checksums at load time
  unless `verifyChecksums(false)` is set. The PyTorch state dict is not hashed at
  load because it is not used for inference.
* `close()` releases native sessions. Call it; do not rely on finalizers.

## Package layout

```
model_dir/
  onnx/encoder.onnx          src_ids, src_pad_mask, position_ids -> memory
  onnx/decoder_init.onnx     tgt_ids, memory, self_mask, cross_mask, position_ids
                             -> logits + self/cross KV cache
  onnx/decoder_step.onnx     tgt_ids, position_ids, past KV -> logits + present KV
  tokenizer/tokenizer.json   special ids, normalization policy, piece table
  tokenizer/merges.txt       merge ranks (exact match with Python)
  model_config.json          architecture, limits, ONNX contract
  generation.json            decoding defaults and suppressed ids
  manifest.json
  parity_vectors.json        Python greedy output the Java tests must match
  torch_state_dict.pt        training-time weights (not used by Java)
```

Masks are additive `float32`; cache tensors are `[batch, heads, sequence, head_dim]`.
All sequence axes are dynamic; the step graph takes exactly one new token.

## Verification

* Full Maven suite: **10 tests, 0 failures, 0 skipped** (2 cache-reorder, 3
  preprocessor, 2 tokenizer-golden, 3 model-package integration) against both
  the step-14,500 and step-15,897 packages.
* Token ids match Python token-for-token on golden fixtures.
* Greedy output matches Python's greedy output token-for-token on both the
  step-14,500 and step-15,897 packages (`TranslatorParityTest`, run with
  `-Dtranscend.model.dir`).
* Beam search reorders the KV cache with the surviving hypotheses
  (`CacheReorderTest`).
* Benchmark on the step-14,500 package: cold load 1,297 ms, first translation
  380 ms, warm median 233 ms, p95 443 ms, batch of 5 inputs 1,219 ms (4.10
  inputs/s). See `reports/java_benchmark.json`.
* Benchmark on the step-15,897 package: cold load 1,189 ms, first translation
  405 ms, warm median 262 ms, p95 331 ms, batch of 5 inputs 1,436 ms (3.48
  inputs/s). See `reports/java_benchmark_step15897.json`.
