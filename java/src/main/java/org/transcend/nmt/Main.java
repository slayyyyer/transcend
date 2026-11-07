package org.transcend.nmt;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** Command-line interface for offline translation, bulk runs and benchmarking. */
public final class Main {
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            usage();
            System.exit(2);
        }
        String command = args[0];
        Map<String, String> options = parseOptions(Arrays.copyOfRange(args, 1, args.length));
        switch (command) {
            case "translate" -> translate(options);
            case "batch" -> batch(options);
            case "bench" -> bench(options);
            case "info" -> info(options);
            default -> {
                usage();
                System.exit(2);
            }
        }
    }

    private static Map<String, String> parseOptions(String[] args) {
        Map<String, String> options = new LinkedHashMap<>();
        List<String> positional = new ArrayList<>();
        for (int index = 0; index < args.length; index++) {
            String arg = args[index];
            if (arg.startsWith("--")) {
                String key = arg.substring(2);
                String value = index + 1 < args.length ? args[++index] : "";
                options.put(key, value);
            } else {
                positional.add(arg);
            }
        }
        options.put("_positional", String.join(" ", positional));
        return options;
    }

    private static void usage() {
        System.out.println("""
                Usage:
                  java -jar transcend-nmt.jar translate --model DIR --from en --to fr "text"
                  java -jar transcend-nmt.jar batch     --model DIR --from en --to fr --file inputs.txt
                  java -jar transcend-nmt.jar bench     --model DIR --from en --to fr [--count 20] [--out report.json]
                  java -jar transcend-nmt.jar info      --model DIR
                """);
    }

    private static Translator openTranslator(Map<String, String> options) throws IOException {
        Path model = Path.of(require(options, "model"));
        RuntimeOptions runtime = RuntimeOptions.builder()
                .intraOpThreads(Integer.parseInt(options.getOrDefault("threads", "0")))
                .maxConcurrency(Integer.parseInt(options.getOrDefault("concurrency", "4")))
                .build();
        return Translator.open(model, runtime);
    }

    private static String require(Map<String, String> options, String key) {
        String value = options.get(key);
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("missing required option --" + key);
        }
        return value;
    }

    private static void translate(Map<String, String> options) throws IOException {
        String text = options.getOrDefault("_positional", "");
        if (text.isBlank()) {
            throw new IllegalArgumentException("no text supplied");
        }
        TranslationOptions.Builder decoding = TranslationOptions.builder();
        if (options.containsKey("beam")) {
            decoding.beamSize(Integer.parseInt(options.get("beam")));
        }
        if (options.containsKey("max-new-tokens")) {
            decoding.maxNewTokens(Integer.parseInt(options.get("max-new-tokens")));
        }
        try (Translator translator = openTranslator(options)) {
            TranslationResult result = translator.translate(
                    text,
                    Language.fromCode(require(options, "from")),
                    Language.fromCode(require(options, "to")),
                    decoding.build());
            System.out.println(result.text());
            System.err.printf(Locale.ROOT, "[model=%s termination=%s source_tokens=%d target_tokens=%d]%n",
                    result.modelId(), result.terminationReason(), result.sourceTokens(), result.targetTokens());
        }
    }

    private static void batch(Map<String, String> options) throws IOException {
        Path input = Path.of(require(options, "file"));
        List<String> lines = Files.readAllLines(input, StandardCharsets.UTF_8);
        try (Translator translator = openTranslator(options)) {
            List<TranslationResult> results = translator.translateBatch(
                    lines,
                    Language.fromCode(require(options, "from")),
                    Language.fromCode(require(options, "to")),
                    TranslationOptions.defaults());
            for (TranslationResult result : results) {
                System.out.println(result.text());
            }
        }
    }

    private static void info(Map<String, String> options) throws IOException {
        try (Translator translator = openTranslator(options)) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("model_id", translator.modelId());
            payload.put("max_source_tokens", translator.maxSourceTokens());
            payload.put("java_version", System.getProperty("java.version"));
            payload.put("os", System.getProperty("os.name") + " " + System.getProperty("os.arch"));
            System.out.println(GSON.toJson(payload));
        }
    }

    private static void bench(Map<String, String> options) throws Exception {
        Path model = Path.of(require(options, "model"));
        Language from = Language.fromCode(require(options, "from"));
        Language to = Language.fromCode(require(options, "to"));
        int count = Integer.parseInt(options.getOrDefault("count", "20"));
        String[] inputs = {
                "The train leaves at eight in the morning.",
                "She works in a small bookshop near the river.",
                "We need two tickets for the concert, please.",
                "The meeting was postponed until Monday because of the storm.",
                "I would like to book a room for three nights."
        };

        long loadStart = System.nanoTime();
        RuntimeOptions runtime = RuntimeOptions.builder().intraOpThreads(
                Integer.parseInt(options.getOrDefault("threads", "0"))).build();
        try (Translator translator = Translator.open(model, runtime)) {
            double loadMs = (System.nanoTime() - loadStart) / 1e6;
            long firstStart = System.nanoTime();
            TranslationResult first = translator.translate(inputs[0], from, to, TranslationOptions.defaults());
            double firstMs = (System.nanoTime() - firstStart) / 1e6;

            List<Double> warm = new ArrayList<>();
            for (int iteration = 0; iteration < count; iteration++) {
                String input = inputs[iteration % inputs.length];
                long start = System.nanoTime();
                translator.translate(input, from, to, TranslationOptions.defaults());
                warm.add((System.nanoTime() - start) / 1e6);
            }
            List<Double> sorted = new ArrayList<>(warm);
            java.util.Collections.sort(sorted);

            long batchStart = System.nanoTime();
            translator.translateBatch(Arrays.asList(inputs), from, to, TranslationOptions.defaults());
            double batchMs = (System.nanoTime() - batchStart) / 1e6;

            Runtime jvm = Runtime.getRuntime();
            long usedHeap = jvm.totalMemory() - jvm.freeMemory();
            Map<String, Object> report = new LinkedHashMap<>();
            report.put("model_id", translator.modelId());
            report.put("java_version", System.getProperty("java.version"));
            report.put("os", System.getProperty("os.name") + " " + System.getProperty("os.arch"));
            report.put("available_processors", jvm.availableProcessors());
            report.put("max_heap_bytes", jvm.maxMemory());
            report.put("used_heap_bytes", usedHeap);
            report.put("cold_load_ms", round(loadMs));
            report.put("first_translation_ms", round(firstMs));
            report.put("first_translation_text", first.text());
            report.put("first_termination", first.terminationReason());
            report.put("warm_runs", warm.size());
            report.put("warm_median_ms", round(median(sorted)));
            report.put("warm_p95_ms", round(percentile(sorted, 0.95)));
            report.put("warm_min_ms", round(sorted.get(0)));
            report.put("warm_max_ms", round(sorted.get(sorted.size() - 1)));
            report.put("batch_5_inputs_ms", round(batchMs));
            report.put("batch_inputs_per_second", round(5.0 / (batchMs / 1000.0)));
            String json = GSON.toJson(report);
            String out = options.get("out");
            if (out != null && !out.isBlank()) {
                Path path = Path.of(out);
                if (path.getParent() != null) {
                    Files.createDirectories(path.getParent());
                }
                Files.writeString(path, json, StandardCharsets.UTF_8);
            }
            System.out.println(json);
        }
    }

    private static double median(List<Double> sorted) {
        int size = sorted.size();
        return size % 2 == 1 ? sorted.get(size / 2) : (sorted.get(size / 2 - 1) + sorted.get(size / 2)) / 2.0;
    }

    private static double percentile(List<Double> sorted, double quantile) {
        int index = (int) Math.round(quantile * (sorted.size() - 1));
        return sorted.get(Math.min(sorted.size() - 1, Math.max(0, index)));
    }

    private static double round(double value) {
        return Math.round(value * 1000.0) / 1000.0;
    }
}
