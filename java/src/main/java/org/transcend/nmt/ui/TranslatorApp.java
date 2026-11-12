package org.transcend.nmt.ui;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import javafx.animation.FadeTransition;
import javafx.animation.PauseTransition;
import javafx.animation.RotateTransition;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.css.PseudoClass;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.OverrunStyle;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.ProgressIndicator;
import javafx.scene.control.TextArea;
import javafx.scene.control.ToggleButton;
import javafx.scene.control.ToggleGroup;
import javafx.scene.control.Tooltip;
import javafx.scene.input.Clipboard;
import javafx.scene.input.ClipboardContent;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.stage.DirectoryChooser;
import javafx.stage.Stage;
import javafx.util.Duration;
import org.transcend.nmt.ByteBpeTokenizer;
import org.transcend.nmt.InputTooLongException;
import org.transcend.nmt.Language;
import org.transcend.nmt.RuntimeOptions;
import org.transcend.nmt.TranslationOptions;
import org.transcend.nmt.TranslationResult;
import org.transcend.nmt.Translator;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * A quiet, offline English&lt;-&gt;French desktop translator built around two
 * facing reading pages.
 */
public final class TranslatorApp extends Application {
    private static final String APP_TITLE = "Transcend Translator";
    private static final String NOTICE_INPUT = "input";
    private static final String NOTICE_MODEL = "model";
    private static final String NOTICE_RUNTIME = "runtime";
    private static final PseudoClass FOCUS_WITHIN = PseudoClass.getPseudoClass("focus-within");
    private static final int DEFAULT_SOURCE_LIMIT = 128;

    private final ExecutorService worker = Executors.newSingleThreadExecutor(runnable -> {
        Thread thread = new Thread(runnable, "transcend-ui-worker");
        thread.setDaemon(true);
        return thread;
    });

    private Stage stage;
    private VBox root;
    private VBox workspace;
    private TextArea sourceArea;
    private TextArea outputArea;
    private Label sourcePlaceholder;
    private Label targetPlaceholder;
    private Label sourceLanguageLabel;
    private Label sourceCodeLabel;
    private Label targetLanguageLabel;
    private Label targetCodeLabel;
    private Label tokenCountLabel;
    private HBox tokenMeterBox;
    private ProgressBar tokenMeter;
    private Button clearButton;
    private Button copyButton;
    private Button swapButton;
    private Button modelButton;
    private Button translateButton;
    private Label translateLabel;
    private Label shortcutLabel;
    private ProgressIndicator translateSpinner;
    private ToggleButton enFrButton;
    private ToggleButton frEnButton;
    private ToggleGroup directionGroup;
    private HBox noticeBar;
    private Label noticeMark;
    private Label noticeText;
    private Region statusDot;
    private Label statusTitleLabel;
    private Label statusDivider;
    private Label statusDetailLabel;
    private PauseTransition tokenDebounce;
    private PauseTransition copyReset;
    private ModelSession session;
    private Path currentModelDirectory;
    private Language sourceLanguage = Language.EN;
    private Language targetLanguage = Language.FR;
    private int tokenLimit = DEFAULT_SOURCE_LIMIT;
    private int tokenCount;
    private boolean overLimit;
    private boolean loadingModel;
    private boolean translating;
    private boolean suppressSourceEvents;
    private boolean suppressDirectionEvents;
    private int loadGeneration;
    private int translationGeneration;
    private String noticeOwner;

    @Override
    public void start(Stage primaryStage) {
        stage = primaryStage;
        tokenDebounce = new PauseTransition(Duration.millis(130));
        tokenDebounce.setOnFinished(event -> updateTokenCount());
        copyReset = new PauseTransition(Duration.millis(1400));
        copyReset.setOnFinished(event -> {
            copyButton.setText("COPY");
            copyButton.getStyleClass().remove("is-copied");
        });

        buildInterface();
        Scene scene = new Scene(root, 1180, 760);
        var stylesheet = TranslatorApp.class.getResource("translator.css");
        if (stylesheet == null) {
            throw new IllegalStateException("missing translator.css");
        }
        scene.getStylesheets().add(stylesheet.toExternalForm());
        installKeyboardShortcuts(scene);

        primaryStage.setTitle(APP_TITLE);
        primaryStage.setMinWidth(940);
        primaryStage.setMinHeight(700);
        primaryStage.setScene(scene);
        primaryStage.show();

        updateShortcutLabel();
        updateDirectionPresentation();
        updateTokenCount();
        setStatus(StatusTone.ERROR, "No model loaded", "Choose an exported ONNX package");
        updateControls();
        Platform.runLater(this::openInitialModel);
    }

    @Override
    public void stop() {
        loadGeneration++;
        translationGeneration++;
        if (tokenDebounce != null) {
            tokenDebounce.stop();
        }
        if (copyReset != null) {
            copyReset.stop();
        }
        worker.shutdownNow();
        try {
            worker.awaitTermination(1500, TimeUnit.MILLISECONDS);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
        if (!translating && session != null) {
            session.translator().close();
            session = null;
        }
    }

    // ---------------------------------------------------------------------
    // Window structure
    // ---------------------------------------------------------------------

    private void buildInterface() {
        VBox sheet = buildSheet();
        HBox notice = buildNotice();
        HBox status = buildStatusbar();
        BorderPane topbar = buildTopbar();

        workspace = new VBox(12);
        workspace.getStyleClass().add("workspace");
        workspace.setAlignment(Pos.CENTER);
        workspace.getChildren().addAll(sheet, notice, status);
        VBox.setVgrow(sheet, Priority.ALWAYS);

        root = new VBox(topbar, workspace);
        root.getStyleClass().add("app");
        VBox.setVgrow(workspace, Priority.ALWAYS);

        wireDirectionToggle();
    }

    private BorderPane buildTopbar() {
        Label mark = new Label("T");
        mark.getStyleClass().add("brand-mark");
        mark.setMinSize(34, 34);
        mark.setPrefSize(34, 34);
        mark.setMaxSize(34, 34);
        mark.setAlignment(Pos.CENTER);

        Label brandName = new Label("transcend");
        brandName.getStyleClass().add("brand-name");
        Label product = new Label("TRANSLATOR");
        product.getStyleClass().add("brand-product");
        HBox brand = new HBox(10, mark, brandName, product);
        brand.setAlignment(Pos.CENTER_LEFT);

        enFrButton = directionButton("EN   →   FR", "English to French");
        frEnButton = directionButton("FR   →   EN", "French to English");
        directionGroup = new ToggleGroup();
        enFrButton.setToggleGroup(directionGroup);
        frEnButton.setToggleGroup(directionGroup);
        enFrButton.setSelected(true);
        HBox directionSwitch = new HBox(2, enFrButton, frEnButton);
        directionSwitch.getStyleClass().add("direction-switch");
        directionSwitch.setAlignment(Pos.CENTER);
        // BorderPane stretches its center node by default; keep the pill snug.
        directionSwitch.setMinSize(Region.USE_PREF_SIZE, Region.USE_PREF_SIZE);
        directionSwitch.setMaxSize(Region.USE_PREF_SIZE, Region.USE_PREF_SIZE);

        Region localDot = new Region();
        localDot.getStyleClass().add("local-dot");
        localDot.setMinSize(6, 6);
        localDot.setPrefSize(6, 6);
        localDot.setMaxSize(6, 6);
        Label localText = new Label("LOCAL");
        localText.getStyleClass().add("local-text");
        HBox local = new HBox(7, localDot, localText);
        local.setAlignment(Pos.CENTER_LEFT);
        local.setAccessibleText("Local model, no network connection");

        modelButton = new Button("MODEL…");
        modelButton.getStyleClass().add("model-button");
        modelButton.setOnAction(event -> chooseModel());
        modelButton.setTooltip(new Tooltip("Choose an exported ONNX model folder"));
        modelButton.setAccessibleText("Choose model folder");

        HBox right = new HBox(16, local, modelButton);
        right.setAlignment(Pos.CENTER_RIGHT);

        BorderPane topbar = new BorderPane();
        topbar.getStyleClass().add("topbar");
        topbar.setMinHeight(82);
        topbar.setPrefHeight(82);
        topbar.setMaxHeight(82);
        topbar.setLeft(brand);
        topbar.setCenter(directionSwitch);
        topbar.setRight(right);
        BorderPane.setAlignment(brand, Pos.CENTER_LEFT);
        BorderPane.setAlignment(directionSwitch, Pos.CENTER);
        BorderPane.setAlignment(right, Pos.CENTER_RIGHT);
        return topbar;
    }

    private ToggleButton directionButton(String text, String accessibleText) {
        ToggleButton button = new ToggleButton(text);
        button.getStyleClass().add("direction-option");
        button.setAccessibleText(accessibleText);
        button.setFocusTraversable(true);
        return button;
    }

    private VBox buildSheet() {
        VBox sourcePage = buildPage(false);
        VBox targetPage = buildPage(true);
        StackPane gutter = buildGutter();
        HBox body = new HBox(sourcePage, gutter, targetPage);
        VBox.setVgrow(body, Priority.ALWAYS);
        HBox.setHgrow(sourcePage, Priority.ALWAYS);
        HBox.setHgrow(targetPage, Priority.ALWAYS);

        Region topRule = new Region();
        topRule.getStyleClass().add("sheet-rule");
        topRule.setMinHeight(3);
        topRule.setPrefHeight(3);
        topRule.setMaxHeight(3);

        VBox sheet = new VBox(topRule, body);
        sheet.getStyleClass().add("sheet");
        sheet.setMinHeight(420);
        VBox.setVgrow(body, Priority.ALWAYS);
        return sheet;
    }

    private VBox buildPage(boolean target) {
        Label folio = new Label(target ? "02" : "01");
        folio.getStyleClass().add("folio");
        folio.setMinSize(22, 22);
        folio.setPrefSize(22, 22);
        folio.setMaxSize(22, 22);
        folio.setAlignment(Pos.CENTER);
        Label role = new Label(target ? "TRANSLATION" : "SOURCE");
        role.getStyleClass().add("page-meta-text");
        HBox meta = new HBox(9, folio, role);
        meta.setAlignment(Pos.CENTER_LEFT);

        Label language = new Label(target ? "French" : "English");
        language.getStyleClass().add("language-name");
        Label code = new Label(target ? "FR" : "EN");
        code.getStyleClass().add("language-code");
        Region languageSpacer = new Region();
        HBox.setHgrow(languageSpacer, Priority.ALWAYS);
        HBox languageRow = new HBox(12, language, languageSpacer, code);
        languageRow.setAlignment(Pos.BOTTOM_LEFT);
        languageRow.getStyleClass().add("language-row");

        if (target) {
            targetLanguageLabel = language;
            targetCodeLabel = code;
            outputArea = createOutputArea();
        } else {
            sourceLanguageLabel = language;
            sourceCodeLabel = code;
            sourceArea = createSourceArea();
        }

        TextArea editor = target ? outputArea : sourceArea;
        Label placeholder = createEditorPlaceholder(editor, target);
        if (target) {
            targetPlaceholder = placeholder;
        } else {
            sourcePlaceholder = placeholder;
        }
        StackPane editorShell = new StackPane(editor, placeholder);
        editorShell.getStyleClass().add("editor-shell");
        editorShell.setAlignment(Pos.TOP_LEFT);
        editor.focusedProperty().addListener((observable, oldValue, focused) ->
                editorShell.pseudoClassStateChanged(FOCUS_WITHIN, focused));
        VBox.setVgrow(editorShell, Priority.ALWAYS);

        HBox footer = target ? buildTargetFooter() : buildSourceFooter();
        footer.getStyleClass().add("page-footer");

        VBox page = new VBox(meta, languageRow, editorShell, footer);
        page.getStyleClass().addAll("page", target ? "target-page" : "source-page");
        page.setMaxHeight(Double.MAX_VALUE);
        page.setFillWidth(true);
        VBox.setMargin(languageRow, new Insets(18, 0, 0, 0));
        VBox.setMargin(editorShell, new Insets(16, 0, 0, 0));
        VBox.setMargin(footer, new Insets(8, 0, 0, 0));
        return page;
    }

    private Label createEditorPlaceholder(TextArea editor, boolean target) {
        Label placeholder = new Label();
        placeholder.getStyleClass().add("editor-placeholder");
        if (target) {
            placeholder.getStyleClass().add("target-placeholder");
        }
        placeholder.setMouseTransparent(true);
        placeholder.setFocusTraversable(false);
        placeholder.setWrapText(true);
        placeholder.setMaxWidth(Double.MAX_VALUE);
        placeholder.setAlignment(Pos.TOP_LEFT);
        placeholder.visibleProperty().bind(Bindings.createBooleanBinding(
                () -> editor.getText().isBlank(), editor.textProperty()));
        StackPane.setAlignment(placeholder, Pos.TOP_LEFT);
        StackPane.setMargin(placeholder, new Insets(8, 18, 2, 4));
        return placeholder;
    }

    private TextArea createSourceArea() {
        TextArea area = new TextArea();
        area.setWrapText(true);
        area.setPrefRowCount(8);
        area.setMinHeight(0);
        area.setMaxHeight(Double.MAX_VALUE);
        area.getStyleClass().addAll("translation-editor", "source-editor");
        area.setAccessibleText("Source text to translate");
        area.textProperty().addListener((observable, oldValue, newValue) -> {
            if (suppressSourceEvents) {
                return;
            }
            if (!outputArea.getText().isBlank()) {
                setOutputText("", false);
            }
            hideNotice(NOTICE_RUNTIME);
            tokenDebounce.playFromStart();
            updateControls();
        });
        return area;
    }

    private TextArea createOutputArea() {
        TextArea area = new TextArea();
        area.setEditable(false);
        area.setWrapText(true);
        area.setPrefRowCount(8);
        area.setMinHeight(0);
        area.setMaxHeight(Double.MAX_VALUE);
        area.setFocusTraversable(true);
        area.getStyleClass().addAll("translation-editor", "target-editor");
        area.setAccessibleText("Translation");
        return area;
    }

    private HBox buildSourceFooter() {
        tokenCountLabel = new Label("0 / " + DEFAULT_SOURCE_LIMIT);
        tokenCountLabel.getStyleClass().add("token-count");
        tokenMeter = new ProgressBar(0);
        tokenMeter.getStyleClass().add("token-meter");
        tokenMeter.setMinWidth(33);
        tokenMeter.setPrefWidth(33);
        tokenMeter.setMaxWidth(33);
        tokenMeter.setFocusTraversable(false);
        Label tokens = new Label("TOKENS");
        tokens.getStyleClass().add("token-unit");
        tokenMeterBox = new HBox(7, tokenCountLabel, tokenMeter, tokens);
        tokenMeterBox.setAlignment(Pos.CENTER_LEFT);
        tokenMeterBox.getStyleClass().add("token-meter-box");

        clearButton = textAction("CLEAR");
        clearButton.setOnAction(event -> clearAll());

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);
        HBox footer = new HBox(12, tokenMeterBox, spacer, clearButton);
        footer.setAlignment(Pos.CENTER_LEFT);
        return footer;
    }

    private HBox buildTargetFooter() {
        Region qualityDot = new Region();
        qualityDot.getStyleClass().add("quality-dot");
        qualityDot.setMinSize(4, 4);
        qualityDot.setPrefSize(4, 4);
        qualityDot.setMaxSize(4, 4);
        Label qualityText = new Label("PILOT QUALITY");
        qualityText.getStyleClass().add("quality-text");
        HBox quality = new HBox(6, qualityDot, qualityText);
        quality.setAlignment(Pos.CENTER_LEFT);
        Tooltip.install(quality, new Tooltip(
                "The local pilot model is useful for gist translation, not production-quality text."));

        copyButton = textAction("COPY");
        copyButton.setOnAction(event -> copyTranslation());

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);
        HBox footer = new HBox(12, quality, spacer, copyButton);
        footer.setAlignment(Pos.CENTER_LEFT);
        return footer;
    }

    private Button textAction(String text) {
        Button button = new Button(text);
        button.getStyleClass().add("text-action");
        button.setFocusTraversable(true);
        return button;
    }

    private StackPane buildGutter() {
        Region seam = new Region();
        seam.getStyleClass().add("seam-line");
        seam.setMinWidth(1);
        seam.setPrefWidth(1);
        seam.setMaxWidth(1);
        seam.setMaxHeight(Double.MAX_VALUE);

        swapButton = new Button("⇄");
        swapButton.getStyleClass().add("swap-button");
        swapButton.setMinSize(44, 44);
        swapButton.setPrefSize(44, 44);
        swapButton.setMaxSize(44, 44);
        swapButton.setAccessibleText("Swap languages and text");
        swapButton.setTooltip(new Tooltip("Swap languages and text"));
        swapButton.setOnAction(event -> swapLanguages());

        StackPane gutter = new StackPane(seam, swapButton);
        gutter.getStyleClass().add("gutter");
        gutter.setMinWidth(68);
        gutter.setPrefWidth(68);
        gutter.setMaxWidth(68);
        StackPane.setAlignment(seam, Pos.CENTER);
        StackPane.setAlignment(swapButton, Pos.TOP_CENTER);
        StackPane.setMargin(swapButton, new Insets(82, 0, 0, 0));
        return gutter;
    }

    private HBox buildNotice() {
        noticeMark = new Label("i");
        noticeMark.getStyleClass().add("notice-mark");
        noticeMark.setMinSize(16, 16);
        noticeMark.setPrefSize(16, 16);
        noticeMark.setMaxSize(16, 16);
        noticeMark.setAlignment(Pos.CENTER);
        noticeText = new Label();
        noticeText.getStyleClass().add("notice-text");
        noticeText.setWrapText(true);
        noticeText.setMaxWidth(Double.MAX_VALUE);
        noticeText.setMinWidth(0);
        Button dismiss = new Button("DISMISS");
        dismiss.getStyleClass().add("notice-dismiss");
        dismiss.setOnAction(event -> hideNotice());

        HBox notice = new HBox(10, noticeMark, noticeText, dismiss);
        notice.getStyleClass().add("notice");
        notice.setAlignment(Pos.CENTER_LEFT);
        notice.setVisible(false);
        notice.setManaged(false);
        HBox.setHgrow(noticeText, Priority.ALWAYS);
        noticeBar = notice;
        return notice;
    }

    private HBox buildStatusbar() {
        statusDot = new Region();
        statusDot.getStyleClass().addAll("status-dot", "status-error");
        statusDot.setMinSize(7, 7);
        statusDot.setPrefSize(7, 7);
        statusDot.setMaxSize(7, 7);

        statusTitleLabel = new Label("NO MODEL LOADED");
        statusTitleLabel.getStyleClass().add("status-title");
        statusDivider = new Label("/");
        statusDivider.getStyleClass().add("status-divider");
        statusDetailLabel = new Label("Choose an exported ONNX package");
        statusDetailLabel.getStyleClass().add("status-detail");
        statusDetailLabel.setTextOverrun(OverrunStyle.ELLIPSIS);
        statusDetailLabel.setMinWidth(0);
        statusDetailLabel.setMaxWidth(390);
        HBox statusCopy = new HBox(8, statusTitleLabel, statusDivider, statusDetailLabel);
        statusCopy.setAlignment(Pos.CENTER_LEFT);
        HBox modelState = new HBox(9, statusDot, statusCopy);
        modelState.setAlignment(Pos.CENTER_LEFT);
        modelState.getStyleClass().add("model-state");

        Label privacy = new Label("Text never leaves this device");
        privacy.getStyleClass().add("privacy");
        Region spacerLeft = new Region();
        Region spacerRight = new Region();
        HBox.setHgrow(spacerLeft, Priority.ALWAYS);
        HBox.setHgrow(spacerRight, Priority.ALWAYS);

        translateButton = new Button();
        translateButton.getStyleClass().add("translate-button");
        translateButton.setMinWidth(184);
        translateButton.setMinHeight(48);
        translateButton.setOnAction(event -> translate());
        translateButton.setAccessibleText("Translate");
        translateLabel = new Label("Translate");
        translateLabel.getStyleClass().add("translate-label");
        shortcutLabel = new Label("Ctrl ↵");
        shortcutLabel.getStyleClass().add("kbd");
        translateSpinner = new ProgressIndicator();
        translateSpinner.getStyleClass().add("translate-spinner");
        translateSpinner.setMinSize(13, 13);
        translateSpinner.setPrefSize(13, 13);
        translateSpinner.setMaxSize(13, 13);
        translateSpinner.setVisible(false);
        translateSpinner.setManaged(false);
        translateSpinner.setFocusTraversable(false);
        HBox translateContent = new HBox(18, translateLabel, translateSpinner, shortcutLabel);
        translateContent.setAlignment(Pos.CENTER);
        translateButton.setGraphic(translateContent);
        translateButton.setTooltip(new Tooltip("Translate (Ctrl+Enter)"));

        HBox statusbar = new HBox(14, modelState, spacerLeft, privacy, spacerRight, translateButton);
        statusbar.getStyleClass().add("statusbar");
        statusbar.setAlignment(Pos.CENTER_LEFT);
        return statusbar;
    }

    private void wireDirectionToggle() {
        directionGroup.selectedToggleProperty().addListener((observable, previous, selected) -> {
            if (suppressDirectionEvents) {
                return;
            }
            if (selected == null) {
                if (previous != null) {
                    previous.setSelected(true);
                }
                return;
            }
            Language nextSource = selected == enFrButton ? Language.EN : Language.FR;
            Language nextTarget = nextSource == Language.EN ? Language.FR : Language.EN;
            if (nextSource == sourceLanguage && nextTarget == targetLanguage) {
                return;
            }
            sourceLanguage = nextSource;
            targetLanguage = nextTarget;
            updateDirectionPresentation();
            setOutputText("", false);
            hideNotice(NOTICE_INPUT);
            hideNotice(NOTICE_RUNTIME);
            tokenDebounce.playFromStart();
            updateControls();
        });
    }

    private void installKeyboardShortcuts(Scene scene) {
        scene.addEventFilter(KeyEvent.KEY_PRESSED, event -> {
            if (event.getCode() == KeyCode.ENTER && (event.isControlDown() || event.isMetaDown())) {
                event.consume();
                if (!translateButton.isDisabled()) {
                    translate();
                }
            } else if (event.getCode() == KeyCode.ESCAPE && noticeBar.isVisible()) {
                event.consume();
                hideNotice();
            }
        });
    }

    private void updateShortcutLabel() {
        boolean mac = System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("mac");
        shortcutLabel.setText(mac ? "⌘ ↵" : "Ctrl ↵");
        translateButton.setAccessibleHelp(mac ? "Keyboard shortcut Command Enter" : "Keyboard shortcut Control Enter");
    }

    // ---------------------------------------------------------------------
    // Model lifecycle
    // ---------------------------------------------------------------------

    private void openInitialModel() {
        Path explicit = explicitModelPath();
        if (explicit != null) {
            loadModel(explicit);
            return;
        }
        Optional<Path> located = ModelLocator.findDefault(Path.of(System.getProperty("user.dir")));
        if (located.isPresent()) {
            loadModel(located.get());
            return;
        }
        setStatus(StatusTone.ERROR, "No model loaded", "Choose an exported ONNX package");
        showNotice(NOTICE_MODEL,
                "No local model was found. Use MODEL… to select an exported ONNX package.", true);
        updateControls();
    }

    private Path explicitModelPath() {
        String named = getParameters().getNamed().get("model");
        if (named != null && !named.isBlank()) {
            return Path.of(named).toAbsolutePath().normalize();
        }
        List<String> raw = getParameters().getRaw();
        for (int index = 0; index < raw.size(); index++) {
            String argument = raw.get(index);
            if (argument.startsWith("--model=") && argument.length() > "--model=".length()) {
                return Path.of(argument.substring("--model=".length())).toAbsolutePath().normalize();
            }
            if ("--model".equals(argument) && index + 1 < raw.size()) {
                return Path.of(raw.get(index + 1)).toAbsolutePath().normalize();
            }
        }
        return null;
    }

    private void chooseModel() {
        DirectoryChooser chooser = new DirectoryChooser();
        chooser.setTitle("Choose a Transcend model package");
        Path initial = currentModelDirectory != null
                ? currentModelDirectory.getParent()
                : ModelLocator.findDefault(Path.of(System.getProperty("user.dir")))
                        .map(Path::getParent)
                        .orElse(Path.of(System.getProperty("user.dir")));
        if (initial != null && Files.isDirectory(initial)) {
            chooser.setInitialDirectory(initial.toFile());
        }
        File selected = chooser.showDialog(stage);
        if (selected != null) {
            loadModel(selected.toPath());
        }
    }

    private void loadModel(Path directory) {
        if (translating || loadingModel) {
            return;
        }
        Path normalized = directory.toAbsolutePath().normalize();
        if (!ModelLocator.isPackage(normalized)) {
            reportModelFailure("That folder is not an exported ONNX model package. Select the folder containing model_config.json, generation.json, manifest.json, onnx/, and tokenizer/.");
            return;
        }

        loadingModel = true;
        int generation = ++loadGeneration;
        hideNotice(NOTICE_MODEL);
        setStatus(StatusTone.BUSY, "Loading local model", normalized.getFileName().toString());
        updateControls();

        CompletableFuture.supplyAsync(() -> {
            try {
                return openSession(normalized);
            } catch (IOException | RuntimeException error) {
                throw new CompletionException(error);
            }
        }, worker).whenComplete((loaded, error) -> Platform.runLater(() -> {
            if (generation != loadGeneration) {
                closeSession(loaded);
                return;
            }
            loadingModel = false;
            if (error != null) {
                String message = readableError(error);
                System.err.println("Model load failed: " + message);
                if (session != null) {
                    setStatus(StatusTone.READY, "Model ready", session.detail());
                } else {
                    setStatus(StatusTone.ERROR, "No model loaded", "Choose an exported ONNX package");
                }
                showNotice(NOTICE_MODEL, "Could not load model: " + message, true);
                updateControls();
                return;
            }

            ModelSession previous = session;
            session = loaded;
            currentModelDirectory = normalized;
            tokenLimit = loaded.translator().maxSourceTokens();
            if (previous != null) {
                previous.translator().close();
            }
            hideNotice(NOTICE_MODEL);
            setStatus(StatusTone.READY, "Model ready", loaded.detail());
            System.out.println("Model ready: " + loaded.detail());
            setOutputText("", false);
            updateTokenCount();
            updateControls();
        }));
    }

    private ModelSession openSession(Path directory) throws IOException {
        Translator opened = Translator.open(directory, RuntimeOptions.builder()
                .intraOpThreads(0)
                .maxConcurrency(1)
                .build());
        try {
            ByteBpeTokenizer tokenizer = ByteBpeTokenizer.load(directory.resolve("tokenizer"));
            return new ModelSession(opened, tokenizer, readModelDetail(directory));
        } catch (IOException | RuntimeException error) {
            opened.close();
            throw error;
        }
    }

    private void reportModelFailure(String message) {
        if (session != null) {
            setStatus(StatusTone.READY, "Model ready", session.detail());
        } else {
            setStatus(StatusTone.ERROR, "No model loaded", "Choose an exported ONNX package");
        }
        showNotice(NOTICE_MODEL, message, true);
        updateControls();
    }

    private void closeSession(ModelSession loaded) {
        if (loaded != null) {
            loaded.translator().close();
        }
    }

    private static String readModelDetail(Path directory) {
        String name = directory.getFileName() == null ? directory.toString() : directory.getFileName().toString();
        try {
            JsonObject config = JsonParser.parseString(Files.readString(
                    directory.resolve("model_config.json"), StandardCharsets.UTF_8)).getAsJsonObject();
            long parameters = config.getAsJsonObject("parameter_count").get("total").getAsLong();
            int step = config.getAsJsonObject("checkpoint").get("step").getAsInt();
            return name + " · " + formatParameterCount(parameters)
                    + " · step " + String.format(Locale.ROOT, "%,d", step);
        } catch (IOException | RuntimeException ignored) {
            return name;
        }
    }

    private static String formatParameterCount(long count) {
        if (count >= 1_000_000L) {
            return String.format(Locale.ROOT, "%.1fM", count / 1_000_000.0);
        }
        if (count >= 1_000L) {
            return String.format(Locale.ROOT, "%.0fK", count / 1_000.0);
        }
        return Long.toString(count);
    }

    // ---------------------------------------------------------------------
    // Translation and counts
    // ---------------------------------------------------------------------

    private void translate() {
        if (translating || loadingModel || session == null) {
            return;
        }
        String input = sourceArea.getText();
        if (input.isBlank()) {
            sourceArea.requestFocus();
            return;
        }

        tokenDebounce.stop();
        updateTokenCount();
        if (overLimit) {
            sourceArea.requestFocus();
            return;
        }

        ModelSession active = session;
        Language from = sourceLanguage;
        Language to = targetLanguage;
        int generation = ++translationGeneration;
        translating = true;
        hideNotice(NOTICE_RUNTIME);
        setStatus(StatusTone.BUSY, "Translating locally", "No network · ONNX Runtime");
        updateControls();

        CompletableFuture.supplyAsync(
                () -> active.translator().translate(input, from, to, TranslationOptions.defaults()),
                worker
        ).whenComplete((result, error) -> Platform.runLater(() -> {
            if (generation != translationGeneration) {
                return;
            }
            translating = false;
            if (error != null) {
                showTranslationFailure(error);
                setStatus(StatusTone.READY, "Model ready", active.detail());
                updateControls();
                return;
            }
            setOutputText(result.text(), true);
            if (!result.completedNaturally()) {
                showNotice(NOTICE_RUNTIME,
                        "The model reached its output limit; the translation may be incomplete.", false);
            } else {
                hideNotice(NOTICE_RUNTIME);
            }
            setStatus(StatusTone.READY, "Model ready", active.detail());
            updateControls();
        }));
    }

    private void showTranslationFailure(Throwable error) {
        Throwable cause = rootCause(error);
        String message;
        if (cause instanceof InputTooLongException tooLong) {
            message = "This text is " + tooLong.tokens() + " source tokens; the model accepts "
                    + tooLong.limit() + ". Shorten it and try again.";
            tokenCount = tooLong.tokens();
            tokenLimit = tooLong.limit();
            overLimit = true;
            tokenCountLabel.setText(tooLong.tokens() + " / " + tooLong.limit());
            tokenMeter.setProgress(1.0);
            tokenMeterBox.getStyleClass().removeAll("is-near", "is-over");
            tokenMeterBox.getStyleClass().add("is-over");
        } else {
            message = "Translation failed: " + readableError(cause);
        }
        showNotice(NOTICE_RUNTIME, message, true);
    }

    private void updateTokenCount() {
        if (tokenCountLabel == null) {
            return;
        }
        if (session == null) {
            tokenCount = 0;
            overLimit = false;
            tokenCountLabel.setText(sourceArea.getText().isBlank()
                    ? "0 / " + tokenLimit
                    : "— / " + tokenLimit);
            tokenMeter.setProgress(0);
            tokenMeterBox.getStyleClass().removeAll("is-near", "is-over");
            updateControls();
            return;
        }

        tokenLimit = session.translator().maxSourceTokens();
        String text = sourceArea.getText();
        boolean exact = true;
        boolean available = true;
        int count;
        if (text.isBlank()) {
            count = 0;
        } else if (text.length() > 8_000) {
            count = tokenLimit + 1;
            exact = false;
        } else {
            try {
                count = session.tokenizer().encode(text, targetLanguage, true).length;
            } catch (RuntimeException error) {
                count = 0;
                exact = false;
                available = false;
            }
        }

        tokenCount = count;
        boolean nowOver = available && count > tokenLimit;
        boolean near = !nowOver && count >= Math.ceil(tokenLimit * 0.8);
        tokenCountLabel.setText(!available
                ? "— / " + tokenLimit
                : exact ? count + " / " + tokenLimit : "> " + tokenLimit);
        tokenMeter.setProgress(available ? Math.min(1.0, (double) count / tokenLimit) : 0);
        tokenMeterBox.getStyleClass().removeAll("is-near", "is-over");
        if (nowOver) {
            tokenMeterBox.getStyleClass().add("is-over");
        } else if (near) {
            tokenMeterBox.getStyleClass().add("is-near");
        }
        tokenCountLabel.setAccessibleText(!available
                ? "Source size is unavailable"
                : exact ? "Source size: " + count + " of " + tokenLimit + " tokens"
                : "Source is over the " + tokenLimit + " token limit");
        tokenMeter.setAccessibleText(tokenCountLabel.getAccessibleText());

        if (nowOver && !overLimit) {
            showNotice(NOTICE_INPUT,
                    (exact ? "This text is " + count + " source tokens" : "This text is over")
                            + "; the model accepts " + tokenLimit + ". Shorten it to translate.",
                    true);
        } else if (!nowOver) {
            hideNotice(NOTICE_INPUT);
        }
        overLimit = nowOver;
        updateControls();
    }

    private void updateControls() {
        if (translateButton == null) {
            return;
        }
        boolean blank = sourceArea.getText().isBlank();
        boolean outputBlank = outputArea.getText().isBlank();
        boolean busy = translating || loadingModel;

        translateButton.setDisable(session == null || translating || loadingModel || blank || overLimit);
        sourceArea.setEditable(!translating);
        enFrButton.setDisable(translating || loadingModel);
        frEnButton.setDisable(translating || loadingModel);
        swapButton.setDisable(translating || loadingModel || (blank && outputBlank));
        clearButton.setDisable(translating || loadingModel || (blank && outputBlank));
        copyButton.setDisable(translating || outputBlank);
        modelButton.setDisable(translating || loadingModel);

        translateLabel.setText(translating ? "Translating" : "Translate");
        translateSpinner.setVisible(translating);
        translateSpinner.setManaged(translating);
        shortcutLabel.setVisible(!translating);
        shortcutLabel.setManaged(!translating);
        translateButton.setAccessibleText(translating ? "Translating" : "Translate");
        if (busy) {
            copyButton.getStyleClass().remove("is-copied");
            copyButton.setText("COPY");
        }
    }

    private void updateDirectionPresentation() {
        String sourceName = languageName(sourceLanguage);
        String targetName = languageName(targetLanguage);
        sourceLanguageLabel.setText(sourceName);
        sourceCodeLabel.setText(sourceLanguage.code().toUpperCase(Locale.ROOT));
        targetLanguageLabel.setText(targetName);
        targetCodeLabel.setText(targetLanguage.code().toUpperCase(Locale.ROOT));
        String sourcePrompt = "Write or paste " + sourceName + " here…";
        String targetPrompt = "Your " + targetName + " translation will appear here.";
        sourcePlaceholder.setText(sourcePrompt);
        targetPlaceholder.setText(targetPrompt);
        sourceArea.setAccessibleText("Source text in " + sourceName);
        sourceArea.setAccessibleHelp(sourcePrompt);
        outputArea.setAccessibleText(targetName + " translation");
        outputArea.setAccessibleHelp(targetPrompt);
        copyButton.setAccessibleText("Copy " + targetName + " translation");
    }

    private static String languageName(Language language) {
        return language == Language.EN ? "English" : "French";
    }

    private void setOutputText(String text, boolean animate) {
        copyReset.stop();
        copyButton.setText("COPY");
        copyButton.getStyleClass().remove("is-copied");
        outputArea.setText(text);
        outputArea.positionCaret(0);
        if (animate && !text.isBlank()) {
            FadeTransition fade = new FadeTransition(Duration.millis(240), outputArea);
            fade.setFromValue(0.25);
            fade.setToValue(1.0);
            fade.play();
        }
        updateControls();
    }

    // ---------------------------------------------------------------------
    // Actions and status
    // ---------------------------------------------------------------------

    private void swapLanguages() {
        if (translating || loadingModel) {
            return;
        }
        String previousSource = sourceArea.getText();
        String previousOutput = outputArea.getText();
        Language oldSource = sourceLanguage;
        sourceLanguage = targetLanguage;
        targetLanguage = oldSource;

        suppressDirectionEvents = true;
        if (sourceLanguage == Language.EN) {
            enFrButton.setSelected(true);
        } else {
            frEnButton.setSelected(true);
        }
        suppressDirectionEvents = false;
        updateDirectionPresentation();

        if (previousOutput.isBlank()) {
            setOutputText("", false);
        } else {
            suppressSourceEvents = true;
            sourceArea.setText(previousOutput);
            suppressSourceEvents = false;
            setOutputText(previousSource, false);
        }
        hideNotice(NOTICE_INPUT);
        hideNotice(NOTICE_RUNTIME);
        tokenDebounce.stop();
        updateTokenCount();
        updateControls();
        sourceArea.requestFocus();

        RotateTransition spin = new RotateTransition(Duration.millis(320), swapButton);
        spin.setByAngle(180);
        spin.play();
    }

    private void clearAll() {
        if (translating || loadingModel) {
            return;
        }
        suppressSourceEvents = true;
        sourceArea.clear();
        suppressSourceEvents = false;
        setOutputText("", false);
        hideNotice(NOTICE_INPUT);
        hideNotice(NOTICE_RUNTIME);
        tokenDebounce.stop();
        updateTokenCount();
        sourceArea.requestFocus();
    }

    private void copyTranslation() {
        String text = outputArea.getText();
        if (text.isBlank()) {
            return;
        }
        ClipboardContent content = new ClipboardContent();
        content.putString(text);
        Clipboard.getSystemClipboard().setContent(content);
        copyButton.setText("COPIED");
        copyButton.getStyleClass().add("is-copied");
        copyReset.playFromStart();
    }

    private void showNotice(String owner, String message, boolean error) {
        noticeOwner = owner;
        noticeText.setText(message);
        noticeBar.getStyleClass().remove("is-error");
        if (error) {
            noticeBar.getStyleClass().add("is-error");
        }
        noticeMark.setText(error ? "!" : "i");
        noticeBar.setAccessibleText(message);
        noticeBar.setVisible(true);
        noticeBar.setManaged(true);
    }

    private void hideNotice(String owner) {
        if (owner.equals(noticeOwner)) {
            hideNotice();
        }
    }

    private void hideNotice() {
        noticeOwner = null;
        noticeBar.setVisible(false);
        noticeBar.setManaged(false);
    }

    private void setStatus(StatusTone tone, String title, String detail) {
        statusTitleLabel.setText(title.toUpperCase(Locale.ROOT));
        boolean hasDetail = detail != null && !detail.isBlank();
        statusDetailLabel.setText(hasDetail ? detail : "");
        statusDetailLabel.setVisible(hasDetail);
        statusDetailLabel.setManaged(hasDetail);
        statusDivider.setVisible(hasDetail);
        statusDivider.setManaged(hasDetail);
        statusTitleLabel.setAccessibleText(title + (hasDetail ? ": " + detail : ""));
        statusDot.getStyleClass().removeAll("status-ready", "status-busy", "status-error");
        statusDot.getStyleClass().add(switch (tone) {
            case READY -> "status-ready";
            case BUSY -> "status-busy";
            case ERROR -> "status-error";
        });
    }

    private static Throwable rootCause(Throwable error) {
        Throwable cause = error;
        while (cause.getCause() != null && cause.getCause() != cause) {
            cause = cause.getCause();
        }
        return cause;
    }

    private static String readableError(Throwable error) {
        Throwable cause = rootCause(error);
        String message = cause.getMessage();
        if (message == null || message.isBlank()) {
            return cause.getClass().getSimpleName();
        }
        return message.replaceAll("\\s+", " ").trim();
    }

    private enum StatusTone {
        READY,
        BUSY,
        ERROR
    }

    private record ModelSession(Translator translator, ByteBpeTokenizer tokenizer, String detail) {
    }
}
