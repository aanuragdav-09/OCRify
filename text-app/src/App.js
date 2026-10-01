import { useEffect, useRef, useState } from "react";
import { createWorker } from "tesseract.js";
import "./App.css";

const wordPattern = /[\p{L}\p{M}\p{N}]+(?:['’][\p{L}\p{M}\p{N}]+)*/gu;

async function preprocessImage(file, selection) {
  const bitmap = await createImageBitmap(file);
  const sourceX = selection ? Math.round(selection.x * bitmap.width) : 0;
  const sourceY = selection ? Math.round(selection.y * bitmap.height) : 0;
  const width = Math.max(1, Math.round((selection?.width ?? 1) * bitmap.width));
  const height = Math.max(1, Math.round((selection?.height ?? 1) * bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });

  if (!context) {
    bitmap.close();
    throw new Error("Image processing is unavailable.");
  }

  context.drawImage(bitmap, sourceX, sourceY, width, height, 0, 0, width, height);
  bitmap.close();

  const imageData = context.getImageData(0, 0, width, height);
  const sourceGray = new Uint8Array(width * height);
  const enhanced = new Uint8Array(width * height);
  const histogram = new Uint32Array(256);
  const pixels = imageData.data;

  for (let index = 0; index < sourceGray.length; index += 1) {
    const offset = index * 4;
    const gray = Math.round(
      pixels[offset] * 0.299 + pixels[offset + 1] * 0.587 + pixels[offset + 2] * 0.114
    );
    sourceGray[index] = gray;
  }

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      const left = sourceGray[y * width + Math.max(0, x - 1)];
      const right = sourceGray[y * width + Math.min(width - 1, x + 1)];
      const above = sourceGray[Math.max(0, y - 1) * width + x];
      const below = sourceGray[Math.min(height - 1, y + 1) * width + x];
      const neighbors = (left + right + above + below) / 4;
      const contrastAdjusted = (sourceGray[index] - 128) * 1.35 + 128;
      const sharpened = Math.max(0, Math.min(255, contrastAdjusted * 1.35 - neighbors * 0.35));
      enhanced[index] = sharpened;
      histogram[Math.round(sharpened)] += 1;
    }
  }

  let totalIntensity = 0;
  for (let intensity = 0; intensity < 256; intensity += 1) {
    totalIntensity += intensity * histogram[intensity];
  }

  let backgroundWeight = 0;
  let backgroundIntensity = 0;
  let threshold = 127;
  let bestVariance = 0;

  for (let intensity = 0; intensity < 256; intensity += 1) {
    backgroundWeight += histogram[intensity];
    if (backgroundWeight === 0) continue;

    const foregroundWeight = enhanced.length - backgroundWeight;
    if (foregroundWeight === 0) break;

    backgroundIntensity += intensity * histogram[intensity];
    const backgroundMean = backgroundIntensity / backgroundWeight;
    const foregroundMean = (totalIntensity - backgroundIntensity) / foregroundWeight;
    const variance = backgroundWeight * foregroundWeight *
      (backgroundMean - foregroundMean) ** 2;

    if (variance > bestVariance) {
      bestVariance = variance;
      threshold = intensity;
    }
  }

  for (let index = 0; index < enhanced.length; index += 1) {
    const value = enhanced[index] <= threshold ? 0 : 255;
    const offset = index * 4;
    pixels[offset] = value;
    pixels[offset + 1] = value;
    pixels[offset + 2] = value;
    pixels[offset + 3] = 255;
  }

  const binary = new Uint8Array(enhanced.length);
  for (let index = 0; index < binary.length; index += 1) binary[index] = pixels[index * 4];
  const cleaned = new Uint8Array(binary);

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = y * width + x;
      let darkNeighbors = 0;
      for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
        for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
          if (offsetX !== 0 || offsetY !== 0) {
            if (binary[(y + offsetY) * width + x + offsetX] === 0) darkNeighbors += 1;
          }
        }
      }
      if (binary[index] === 0 && darkNeighbors <= 1) cleaned[index] = 255;
      if (binary[index] === 255 && darkNeighbors >= 7) cleaned[index] = 0;
    }
  }

  for (let index = 0; index < cleaned.length; index += 1) {
    const offset = index * 4;
    pixels[offset] = cleaned[index];
    pixels[offset + 1] = cleaned[index];
    pixels[offset + 2] = cleaned[index];
  }
  context.putImageData(imageData, 0, 0);

  return { canvas, sourceGray, width, height };
}

function getOcrBlocks(data) {
  const blocks = (data.blocks ?? []).map((block) => ({
    paragraphs: (block.paragraphs ?? []).map((paragraph) => paragraph.lines ?? []),
  })).filter((block) => block.paragraphs.some((paragraph) => paragraph.length));

  if (blocks.length) return blocks;

  const lines = data.lines ?? (data.words ?? []).map((word) => ({ words: [word] }));
  return lines.length ? [{ paragraphs: [lines] }] : [];
}

function getWordBox(word) {
  return word.bbox ?? { x0: 0, y0: 0, x1: 0, y1: 0 };
}

function getBoxContrast(gray, width, height, box) {
  const left = Math.max(0, Math.floor(box.x0));
  const top = Math.max(0, Math.floor(box.y0));
  const right = Math.min(width, Math.ceil(box.x1));
  const bottom = Math.min(height, Math.ceil(box.y1));
  let minimum = 255;
  let maximum = 0;

  for (let y = top; y < bottom; y += 2) {
    for (let x = left; x < right; x += 2) {
      const value = gray[y * width + x];
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
  }

  return maximum - minimum;
}

function filterOcrData(data, image, options) {
  const blocks = getOcrBlocks(data);
  const preparedBlocks = blocks.map((block) => {
    const paragraphs = block.paragraphs.map((lines) => lines.map((line) => {
      const words = (line.words ?? []).filter((word) => {
        const value = (word.text ?? "").trim();
        const box = getWordBox(word);
        const width = box.x1 - box.x0;
        const height = box.y1 - box.y0;
        const characters = Array.from(value);
        const specialCount = characters.filter((character) =>
          !/[\p{L}\p{M}\p{N}\s]/u.test(character)
        ).length;
        const hasValidSingleLetter = /^[ai]$/iu.test(value);

        if (!value || word.confidence < options.confidence) return false;
        if (options.removeOcrNoise && characters.length === 1 &&
          !hasValidSingleLetter && !/\p{N}/u.test(value)) return false;
        if (options.removeOcrNoise && characters.length &&
          specialCount / characters.length > 0.5) return false;
        if (options.ignoreSmall && (height < 12 || width < 20)) return false;

        const centerX = (box.x0 + box.x1) / 2 / image.width;
        const centerY = (box.y0 + box.y1) / 2 / image.height;
        if (options.ignoreBackground &&
          (centerX < 0.025 || centerX > 0.975 || centerY < 0.025 || centerY > 0.975)) {
          return false;
        }
        if (options.ignoreWatermarks &&
          getBoxContrast(image.sourceGray, image.width, image.height, box) < 32) {
          return false;
        }

        return true;
      });
      const lineText = words.map((word) => word.text.trim()).join(" ");
      const letters = Array.from(lineText).filter((character) => /\p{L}/u.test(character));
      const specialCount = Array.from(lineText).filter((character) =>
        !/[\p{L}\p{M}\p{N}\s]/u.test(character)
      ).length;
      const lineBox = words.reduce((box, word) => {
        const wordBox = getWordBox(word);
        return {
          x0: Math.min(box.x0, wordBox.x0),
          y0: Math.min(box.y0, wordBox.y0),
          x1: Math.max(box.x1, wordBox.x1),
          y1: Math.max(box.y1, wordBox.y1),
        };
      }, { x0: image.width, y0: image.height, x1: 0, y1: 0 });
      const score = words.reduce((total, word) => {
        const box = getWordBox(word);
        const letterCount = Array.from(word.text).filter((character) => /[\p{L}\p{N}]/u.test(character)).length;
        return total + letterCount * word.confidence * Math.sqrt(Math.max(1, box.y1 - box.y0));
      }, 0);

      return {
        text: options.removeOcrNoise &&
          specialCount / Math.max(1, Array.from(lineText).length) > 0.5 ? "" : lineText,
        words,
        box: lineBox,
        score,
        letters,
      };
    }));
    const score = paragraphs.flat().reduce((total, line) => total + line.score, 0);
    return { paragraphs, score };
  });

  const highestBlockScore = Math.max(0, ...preparedBlocks.map((block) => block.score));
  const retainedLines = [];
  const retainedWords = [];

  preparedBlocks.forEach((block) => {
    if (options.mainTextOnly && highestBlockScore > 0 && block.score < highestBlockScore * 0.04) return;

    block.paragraphs.forEach((paragraph) => {
      const paragraphLines = paragraph.filter((line) => {
        if (!line.text) return false;
        if (options.ignoreDecorativeText && line.words.length <= 2 &&
          line.letters.length <= 5 && line.box.y1 - line.box.y0 < 18) {
          return false;
        }
        if (options.ignoreLogos && line.words.length === 1 && line.letters.length <= 5 &&
          line.letters.length > 1 && line.letters.every((letter) =>
            letter === letter.toUpperCase() && letter !== letter.toLowerCase()
          ) && (line.box.y0 < image.height * 0.08 || line.box.y1 > image.height * 0.92)) {
          return false;
        }
        return true;
      });

      if (paragraphLines.length) {
        retainedLines.push(paragraphLines.map((line) => line.text).join("\n"));
        paragraphLines.forEach((line) => retainedWords.push(...line.words));
      }
    });
  });

  const text = retainedLines.join("\n\n");
  const confidence = retainedWords.length
    ? Math.round(retainedWords.reduce((total, word) => total + word.confidence, 0) / retainedWords.length)
    : 0;
  const detectedWords = blocks.reduce((total, block) => total +
    block.paragraphs.reduce((paragraphTotal, paragraph) => paragraphTotal +
      paragraph.reduce((lineTotal, line) => lineTotal + (line.words?.length ?? 0), 0), 0), 0);

  return { text, confidence, retainedWords: retainedWords.length, detectedWords };
}

function getWords(value) {
  return value.match(wordPattern) ?? [];
}

function getEditDistance(firstValue, secondValue) {
  const first = Array.from(firstValue);
  const second = Array.from(secondValue);

  if (first.length === 0) return second.length;
  if (second.length === 0) return first.length;

  let previousRow = Array.from({ length: second.length + 1 }, (_, index) => index);

  for (let row = 1; row <= first.length; row += 1) {
    const currentRow = [row];

    for (let column = 1; column <= second.length; column += 1) {
      const substitutionCost = first[row - 1] === second[column - 1] ? 0 : 1;
      currentRow[column] = Math.min(
        currentRow[column - 1] + 1,
        previousRow[column] + 1,
        previousRow[column - 1] + substitutionCost
      );
    }

    previousRow = currentRow;
  }

  return previousRow[second.length];
}

function getStatistics(value) {
  const characters = Array.from(value);
  const words = getWords(value);
  const letters = characters.filter((character) => /\p{L}/u.test(character));
  const paragraphs = value
    .split(/(?:\r\n|\r|\n)(?:[ \t]*(?:\r\n|\r|\n))+/u)
    .filter((paragraph) => paragraph.trim());
  const sentences = value.trim()
    ? value.trim().split(/[.!?]+/u).filter((sentence) => sentence.trim())
    : [];
  const isUppercase = (character) =>
    character === character.toUpperCase() && character !== character.toLowerCase();
  const isLowercase = (character) =>
    character === character.toLowerCase() && character !== character.toUpperCase();
  const capitalizedWords = words.filter((word) => {
    const wordLetters = Array.from(word).filter((character) => /\p{L}/u.test(character));
    return wordLetters.length > 0 && isUppercase(wordLetters[0]) &&
      wordLetters.slice(1).every(isLowercase);
  });
  const mixedCaseWords = words.filter((word) => {
    const wordLetters = Array.from(word).filter((character) => /\p{L}/u.test(character));
    const hasUppercase = wordLetters.some(isUppercase);
    const hasLowercase = wordLetters.some(isLowercase);
    const isCapitalized = wordLetters.length > 0 && isUppercase(wordLetters[0]) &&
      wordLetters.slice(1).every(isLowercase);
    return hasUppercase && hasLowercase && !isCapitalized;
  });

  return {
    characters: characters.length,
    charactersWithoutSpaces: characters.filter((character) => !/\s/u.test(character)).length,
    words: words.length,
    lines: value.length === 0 ? 0 : value.split(/\r\n|\r|\n/u).length,
    sentences: sentences.length,
    paragraphs: paragraphs.length,
    numbers: value.match(/\p{N}+(?:[.,]\p{N}+)*/gu)?.length ?? 0,
    specialCharacters: characters.filter((character) =>
      !/[\p{L}\p{M}\p{N}\s]/u.test(character)
    ).length,
    uppercaseCharacters: letters.filter(isUppercase).length,
    lowercaseCharacters: letters.filter(isLowercase).length,
    capitalizedWords: capitalizedWords.length,
    mixedCaseWords: mixedCaseWords.length,
  };
}

function compareText(reference, extracted) {
  const referenceCharacters = Array.from(reference);
  const extractedCharacters = Array.from(extracted);
  const longestText = Math.max(referenceCharacters.length, extractedCharacters.length);
  const characterMatch = longestText === 0
    ? 100
    : Math.round((1 - getEditDistance(reference, extracted) / longestText) * 100);
  const referenceWords = getWords(reference);
  const extractedWords = getWords(extracted);
  const availableWords = new Map();

  extractedWords.forEach((word) => {
    const normalizedWord = word.toLocaleLowerCase();
    availableWords.set(normalizedWord, (availableWords.get(normalizedWord) ?? 0) + 1);
  });

  let matchedWords = 0;
  const missingCandidates = [];

  referenceWords.forEach((word) => {
    const normalizedWord = word.toLocaleLowerCase();
    const availableCount = availableWords.get(normalizedWord) ?? 0;

    if (availableCount > 0) {
      matchedWords += 1;
      availableWords.set(normalizedWord, availableCount - 1);
    } else {
      missingCandidates.push(word);
    }
  });

  const extraCandidates = [];
  extractedWords.forEach((word) => {
    const normalizedWord = word.toLocaleLowerCase();
    const availableCount = availableWords.get(normalizedWord) ?? 0;

    if (availableCount > 0) {
      extraCandidates.push(word);
      availableWords.set(normalizedWord, availableCount - 1);
    }
  });

  const wordDenominator = Math.max(referenceWords.length, extractedWords.length);
  const spellingDifferences = [];
  const missingWords = [];
  const usedExtras = new Set();

  missingCandidates.forEach((missingWord) => {
    let closestIndex = -1;
    let closestDistance = Number.POSITIVE_INFINITY;

    extraCandidates.forEach((extraWord, index) => {
      if (usedExtras.has(index)) return;
      const distance = getEditDistance(
        missingWord.toLocaleLowerCase(),
        extraWord.toLocaleLowerCase()
      );
      const allowedDistance = Math.max(1, Math.floor(
        Math.max(missingWord.length, extraWord.length) * 0.4
      ));

      if (distance <= allowedDistance && distance < closestDistance) {
        closestIndex = index;
        closestDistance = distance;
      }
    });

    if (closestIndex === -1) {
      missingWords.push(missingWord);
    } else {
      usedExtras.add(closestIndex);
      spellingDifferences.push({
        reference: missingWord,
        extracted: extraCandidates[closestIndex],
      });
    }
  });

  return {
    characterMatch: Math.max(0, characterMatch),
    wordMatch: wordDenominator === 0
      ? 100
      : Math.round((matchedWords / wordDenominator) * 100),
    missingWords,
    extraWords: extraCandidates.filter((_, index) => !usedExtras.has(index)),
    spellingDifferences,
  };
}

function downloadFile(contents, fileName, mimeType) {
  const file = new Blob([contents], { type: mimeType });
  const url = URL.createObjectURL(file);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  link.click();
  URL.revokeObjectURL(url);
}

function TextConverter() {
  const [text, setText] = useState("");
  const [referenceText, setReferenceText] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("");
  const [ocrConfidence, setOcrConfidence] = useState(0);
  const [imageFile, setImageFile] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState("");
  const [selection, setSelection] = useState(null);
  const [dragStart, setDragStart] = useState(null);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const fileInputRef = useRef(null);
  const rawOcrData = useRef(null);
  const processedImageData = useRef(null);
  const [ocrOptions, setOcrOptions] = useState({
    confidence: 85,
    mainTextOnly: true,
    ignoreBackground: true,
    ignoreWatermarks: true,
    ignoreSmall: true,
    ignoreLogos: true,
    ignoreDecorativeText: true,
    removeOcrNoise: true,
    selectArea: false,
  });
  const statistics = getStatistics(text);
  const comparison = referenceText.trim() ? compareText(referenceText, text) : null;

  const handleUploadClick = () => fileInputRef.current?.click();

  const closeMenu = () => setIsMenuOpen(false);

  const applyOcrFilters = (data, image, options) => {
    const filtered = filterOcrData(data, image, options);
    setText(filtered.text);
    setOcrConfidence(filtered.confidence);
    setMessage(filtered.text
      ? `Kept ${filtered.retainedWords} of ${filtered.detectedWords} detected words. Review the OCR result for mistakes.`
      : `No text passed the current filters at ${options.confidence}% confidence. Lower the slider or disable a filter.`);
  };

  useEffect(() => () => {
    if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
  }, [imagePreviewUrl]);

  const runOcr = async (image, area = null) => {
    if (!image) return;
    setIsProcessing(true);
    setProgress(0);
    setMessage("Preprocessing image...");

    let worker;

    try {
      const processedImage = await preprocessImage(image, area);
      worker = await createWorker("eng", 1, {
        logger: (status) => {
          if (status.status === "recognizing text") {
            setProgress(Math.round(status.progress * 100));
            setMessage(`Recognizing and filtering text... ${Math.round(status.progress * 100)}%`);
          }
        },
      });

      const { data } = await worker.recognize(processedImage.canvas, {}, { blocks: true });
      rawOcrData.current = data;
      processedImageData.current = processedImage;
      applyOcrFilters(data, processedImage, ocrOptions);
    } catch {
      setMessage("Could not process this image. Please try another one.");
    } finally {
      if (worker) await worker.terminate();
      setIsProcessing(false);
    }
  };

  const handleImageUpload = async (event) => {
    const image = event.target.files?.[0];
    event.target.value = "";

    if (!image) return;
    if (!image.type.startsWith("image/")) {
      setMessage("Choose an image file to extract text.");
      return;
    }

    setImageFile(image);
    setImagePreviewUrl(URL.createObjectURL(image));
    setSelection(null);
    await runOcr(image);
  };

  const updateOption = (event) => {
    const { name, checked, value, type } = event.target;
    const nextOptions = {
      ...ocrOptions,
      [name]: type === "checkbox" ? checked : Number(value),
    };
    setOcrOptions(nextOptions);

    if (rawOcrData.current && processedImageData.current) {
      applyOcrFilters(rawOcrData.current, processedImageData.current, nextOptions);
    }
  };

  const getAreaPoint = (event) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
      y: Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height)),
    };
  };

  const handleAreaPointerDown = (event) => {
    if (!imagePreviewUrl) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = getAreaPoint(event);
    setDragStart(point);
    setSelection({ x: point.x, y: point.y, width: 0, height: 0 });
  };

  const handleAreaPointerMove = (event) => {
    if (!dragStart) return;
    const point = getAreaPoint(event);
    setSelection({
      x: Math.min(dragStart.x, point.x),
      y: Math.min(dragStart.y, point.y),
      width: Math.abs(point.x - dragStart.x),
      height: Math.abs(point.y - dragStart.y),
    });
  };

  const handleAreaPointerUp = (event) => {
    if (!dragStart) return;
    handleAreaPointerMove(event);
    setDragStart(null);
  };

  const handleCapitalize = () => {
    setText(text.toLowerCase().replace(
      /(^|[^\p{L}\p{N}])([\p{L}\p{M}])/gu,
      (_, separator, letter) => `${separator}${letter.toUpperCase()}`
    ));
  };

  const handleRemoveExtraSpaces = () => {
    setText(text.replace(/[ \t]{2,}/gu, " ").replace(/^[ \t]+|[ \t]+$/gmu, ""));
  };

  const handleRemoveLineBreaks = () => {
    setText(text.replace(/(?:\r\n|\r|\n)+/gu, " ").replace(/[ \t]{2,}/gu, " ").trim());
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setMessage("Text copied to clipboard.");
    } catch {
      setMessage("Clipboard access is unavailable in this browser.");
    }
  };

  const handleDownloadCsv = () => {
    const escapedText = text.replace(/"/gu, '""');
    downloadFile(`"Extracted Text"\r\n"${escapedText}"`, "extracted-text.csv", "text/csv;charset=utf-8");
  };

  const handleClear = () => {
    setText("");
    setMessage("Text cleared.");
  };

  const actions = [
    { label: "Copy Text", onClick: handleCopy },
    { label: "Clear Text", onClick: handleClear },
    { label: "Download TXT", onClick: () => downloadFile(text, "extracted-text.txt", "text/plain;charset=utf-8") },
    { label: "Download CSV", onClick: handleDownloadCsv },
    { label: "Convert to UPPERCASE", onClick: () => setText(text.toUpperCase()) },
    { label: "Convert to lowercase", onClick: () => setText(text.toLowerCase()) },
    { label: "Capitalize Case", onClick: handleCapitalize },
    { label: "Remove Extra Spaces", onClick: handleRemoveExtraSpaces },
    { label: "Remove Line Breaks", onClick: handleRemoveLineBreaks },
    { label: "Reverse Text", onClick: () => setText(Array.from(text).reverse().join("")) },
  ];

  const statisticItems = [
    ["Total Characters", statistics.characters],
    ["Characters (excluding spaces)", statistics.charactersWithoutSpaces],
    ["Total Words", statistics.words],
    ["Total Lines", statistics.lines],
    ["Total Sentences", statistics.sentences],
    ["Total Paragraphs", statistics.paragraphs],
    ["Total Numbers", statistics.numbers],
    ["Total Special Characters", statistics.specialCharacters],
  ];

  const caseItems = [
    ["UPPERCASE Characters", statistics.uppercaseCharacters],
    ["lowercase Characters", statistics.lowercaseCharacters],
    ["Capitalized Words", statistics.capitalizedWords],
    ["Mixed Case Words", statistics.mixedCaseWords],
  ];
  const hasValidSelection = Boolean(selection && selection.width >= 0.01 && selection.height >= 0.01);

  return (
    <div className={`site-shell${isDarkMode ? " theme-dark" : ""}`} id="top">
      <header className="site-header">
        <nav className="site-nav" aria-label="Main navigation">
          <a className="site-brand" href="#top" aria-label="OCRify home" onClick={closeMenu}>
            <span className="brand-mark" aria-hidden="true">OCR</span>
            <span>OCRify</span>
          </a>
          <button
            type="button"
            className="menu-toggle"
            aria-label={isMenuOpen ? "Close navigation" : "Open navigation"}
            aria-expanded={isMenuOpen}
            aria-controls="primary-links"
            onClick={() => setIsMenuOpen((open) => !open)}
          >
            <span />
            <span />
            <span />
          </button>
          <div className={`nav-content${isMenuOpen ? " is-open" : ""}`}>
            <div className="nav-links" id="primary-links">
              <a href="#top" onClick={closeMenu}>Home</a>
              <a href="#ocr-extractor" onClick={closeMenu}>OCR Extractor</a>
              <a href="#comparison-heading" onClick={closeMenu}>Text Compare</a>
              <a href="#word-counter" onClick={closeMenu}>Word Counter</a>
              <a href="#features" onClick={closeMenu}>Features</a>
              <a href="#faq" onClick={closeMenu}>FAQ</a>
            </div>
            <label className="theme-toggle">
              <input
                type="checkbox"
                checked={isDarkMode}
                onChange={(event) => setIsDarkMode(event.target.checked)}
              />
              <span className="theme-switch" aria-hidden="true" />
              <span>Dark mode</span>
            </label>
          </div>
        </nav>
      </header>

      <main className="text-converter">
        <section className="hero-section" aria-labelledby="hero-title">
          <div className="hero-copy">
            <p className="eyebrow">OCR, WITHOUT THE BUSYWORK</p>
            <h1 id="hero-title">Extract Text From Images Instantly</h1>
            <p className="intro">
              Turn photos, scans, and printed pages into editable text. Review the result,
              compare it, and get useful counts in one place.
            </p>
            <div className="hero-actions">
              <button type="button" className="hero-button primary" onClick={handleUploadClick}>
                Upload Image
              </button>
              <a className="hero-button secondary" href="#ocr-extractor">Try OCR Now</a>
            </div>
          </div>
          <div className="hero-visual" aria-hidden="true">
            <div className="visual-paper">
              <span className="paper-label">IMAGE INPUT</span>
              <span className="paper-line long" />
              <span className="paper-line medium" />
              <span className="paper-line short" />
              <span className="paper-rule" />
              <span className="paper-label extracted-label">EXTRACTED TEXT</span>
              <span className="paper-line medium dark-line" />
              <span className="paper-line long dark-line" />
              <span className="paper-line short dark-line" />
              <span className="visual-spark">Aa</span>
            </div>
            <span className="visual-caption">From pixels to editable words</span>
          </div>
          <a className="hero-next" href="#ocr-extractor">Scroll to extractor <span aria-hidden="true">↓</span></a>
        </section>

        <section className="upload-section" id="ocr-extractor" aria-labelledby="upload-heading">
        <div>
          <h2 id="upload-heading">Upload an image</h2>
          <p>English OCR runs in your browser. Only the extracted text appears in the editor.</p>
        </div>
        <label className="file-picker">
          <span>{isProcessing ? `Reading image... ${progress}%` : "Choose image"}</span>
          <input ref={fileInputRef} type="file" accept="image/*" onChange={handleImageUpload} disabled={isProcessing} />
        </label>
        </section>

        <section className="ocr-settings" aria-labelledby="settings-heading">
        <div className="settings-heading">
          <h2 id="settings-heading">OCR Filters</h2>
          <span>Adjust what is kept in the extracted text</span>
        </div>
        <label className="confidence-control" htmlFor="confidence-range">
          <span>Minimum confidence</span>
          <strong>{ocrOptions.confidence}%</strong>
          <input
            id="confidence-range"
            name="confidence"
            type="range"
            min="50"
            max="100"
            value={ocrOptions.confidence}
            onChange={updateOption}
          />
        </label>
        <div className="filter-options">
          {[
            ["mainTextOnly", "Main text only"],
            ["ignoreBackground", "Ignore background text"],
            ["ignoreWatermarks", "Ignore watermarks (low-contrast text)"],
            ["ignoreSmall", "Ignore text under 12px high / 20px wide"],
            ["ignoreDecorativeText", "Ignore small decorative text"],
            ["ignoreLogos", "Ignore likely logo text"],
            ["removeOcrNoise", "Remove OCR noise"],
            ["selectArea", "Select OCR area"],
          ].map(([name, label]) => (
            <label className="filter-option" key={name}>
              <input
                type="checkbox"
                name={name}
                checked={ocrOptions[name]}
                onChange={updateOption}
              />
              <span>{label}</span>
            </label>
          ))}
        </div>
        <p className="filter-note">
          Background, watermark, logo, and decorative-text detection uses image heuristics and may omit valid text.
        </p>
        </section>

        {imagePreviewUrl && ocrOptions.selectArea && (
        <section className="area-tool" aria-labelledby="area-heading">
          <div className="settings-heading">
            <h2 id="area-heading">Select OCR Area</h2>
            <button type="button" className="text-button" onClick={() => setSelection(null)}>
              Clear selection
            </button>
          </div>
          <p>Drag over the image area you want OCR to read. The source image is shown only while this tool is enabled.</p>
          <div
            className="selection-stage"
            onPointerDown={handleAreaPointerDown}
            onPointerMove={handleAreaPointerMove}
            onPointerUp={handleAreaPointerUp}
            onPointerCancel={handleAreaPointerUp}
          >
            <img src={imagePreviewUrl} alt="Source for choosing an OCR area" draggable="false" />
            {selection && (
              <div
                className="selection-box"
                style={{
                  left: `${selection.x * 100}%`,
                  top: `${selection.y * 100}%`,
                  width: `${selection.width * 100}%`,
                  height: `${selection.height * 100}%`,
                }}
              />
            )}
          </div>
        </section>
        )}

        {imageFile && (
        <div className="ocr-run-row">
          <button
            type="button"
            className="run-ocr-button"
            onClick={() => runOcr(imageFile, ocrOptions.selectArea ? selection : null)}
            disabled={isProcessing || (ocrOptions.selectArea && !hasValidSelection)}
          >
            {isProcessing ? `Processing ${progress}%` : "Run OCR again"}
          </button>
        </div>
        )}
      {isProcessing && <progress className="ocr-progress" max="100" value={progress} />}
      <p className="status-message" aria-live="polite">{message}</p>

        <section className="editor-section" aria-labelledby="extracted-heading">
        <div className="section-heading">
          <h2 id="extracted-heading">Extracted Text</h2>
          <span>{statistics.characters} characters · OCR confidence {ocrConfidence}%</span>
        </div>
        <textarea
          aria-label="Extracted text"
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Upload an image or enter text here..."
          spellCheck="false"
        />
        </section>

        <section className="actions-section" aria-labelledby="actions-heading">
        <h2 id="actions-heading">Quick Actions</h2>
        <div className="action-list">
          {actions.map((action) => (
            <button key={action.label} type="button" onClick={action.onClick} disabled={!text}>
              {action.label}
            </button>
          ))}
        </div>
        </section>

        <section className="statistics-section" id="word-counter" aria-labelledby="statistics-heading">
        <h2 id="statistics-heading">Text Statistics</h2>
        <dl className="statistics-grid">
          {statisticItems.map(([label, value]) => (
            <div className="statistic-item" key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
        </section>

        <section className="statistics-section" aria-labelledby="case-heading">
        <h2 id="case-heading">Text Case Analysis</h2>
        <dl className="statistics-grid case-grid">
          {caseItems.map(([label, value]) => (
            <div className="statistic-item" key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
        </section>

        <section className="comparison-section" aria-labelledby="comparison-heading">
        <h2 id="comparison-heading">Accuracy Analysis</h2>
        <label className="reference-label" htmlFor="reference-text">
          Reference Text
        </label>
        <textarea
          id="reference-text"
          value={referenceText}
          onChange={(event) => setReferenceText(event.target.value)}
          placeholder="Paste the original text to compare it with the extracted text..."
          spellCheck="false"
        />
        {comparison ? (
          <>
            <dl className="match-summary">
              <div><dt>Character Match</dt><dd>{comparison.characterMatch}%</dd></div>
              <div><dt>Word Match</dt><dd>{comparison.wordMatch}%</dd></div>
            </dl>
            <div className="difference-grid">
              <div className="difference-item missing">
                <h3>Missing Words</h3>
                <p>{comparison.missingWords.join(", ") || "None"}</p>
              </div>
              <div className="difference-item extra">
                <h3>Extra Words</h3>
                <p>{comparison.extraWords.join(", ") || "None"}</p>
              </div>
              <div className="difference-item spelling">
                <h3>Spelling Differences</h3>
                {comparison.spellingDifferences.length ? (
                  <ul>
                    {comparison.spellingDifferences.map((difference, index) => (
                      <li key={`${difference.reference}-${difference.extracted}-${index}`}>
                        {difference.reference} → {difference.extracted}
                      </li>
                    ))}
                  </ul>
                ) : <p>None</p>}
              </div>
            </div>
          </>
        ) : (
          <p className="comparison-hint">Enter reference text to see the match analysis.</p>
        )}
        </section>

        <section className="features-section" id="features" aria-labelledby="features-heading">
          <div className="lower-section-heading">
            <p className="eyebrow">A CLEARER WORKFLOW</p>
            <h2 id="features-heading">Everything after the scan</h2>
          </div>
          <div className="feature-list">
            <article className="feature-item">
              <span className="feature-index">01</span>
              <h3>Clean extraction</h3>
              <p>Filter low-confidence noise and edit the recognized text directly.</p>
            </article>
            <article className="feature-item">
              <span className="feature-index">02</span>
              <h3>Useful counts</h3>
              <p>Track words, characters, lines, and text case as you make changes.</p>
            </article>
            <article className="feature-item">
              <span className="feature-index">03</span>
              <h3>Quick comparison</h3>
              <p>Compare OCR output with a reference and spot missing or extra words.</p>
            </article>
          </div>
        </section>

        <section className="faq-section" id="faq" aria-labelledby="faq-heading">
          <div className="lower-section-heading">
            <p className="eyebrow">GOOD TO KNOW</p>
            <h2 id="faq-heading">Frequently asked questions</h2>
          </div>
          <details>
            <summary>What image types can I upload?</summary>
            <p>Choose a common browser-supported image format such as PNG, JPG, or WebP.</p>
          </details>
          <details>
            <summary>Can I edit the extracted text?</summary>
            <p>Yes. The result stays editable, and the statistics update as you make changes.</p>
          </details>
          <details>
            <summary>Does OCR always recognize everything correctly?</summary>
            <p>No. Image quality, font size, and contrast affect recognition. Review the output before using it.</p>
          </details>
        </section>
      </main>

      <footer className="site-footer">
        <div className="footer-content">
          <div className="footer-brand">
            <a className="site-brand" href="#top">
              <span className="brand-mark" aria-hidden="true">OCR</span>
              <span>OCRify</span>
            </a>
            <p>Make the words in your images ready to read, edit, and use.</p>
            <p className="privacy-note">Images are processed in your browser.</p>
          </div>
          <div className="footer-column">
            <h2>Quick Links</h2>
            <a href="#top">Home</a>
            <a href="#features">Features</a>
            <a href="#faq">FAQ</a>
          </div>
          <div className="footer-column">
            <h2>OCR Tools</h2>
            <a href="#ocr-extractor">Image to text</a>
            <a href="#comparison-heading">Text compare</a>
            <a href="#word-counter">Word counter</a>
          </div>
          <div className="footer-column">
            <h2>Contact</h2>
            <a href="mailto:hello@ocrify.app">Email OCRify</a>
            <div className="social-links" aria-label="Social media">
              <a href="https://www.linkedin.com/" aria-label="LinkedIn">in</a>
              <a href="https://www.instagram.com/" aria-label="Instagram">◎</a>
              <a href="https://x.com/" aria-label="X">X</a>
            </div>
          </div>
          <div className="footer-legal">
            <details className="legal-detail" id="privacy">
              <summary>Privacy Policy</summary>
              <p>Your selected image is processed in this browser and is not uploaded to an OCRify server. OCR language data may be downloaded when needed.</p>
            </details>
            <details className="legal-detail" id="terms">
              <summary>Terms &amp; Conditions</summary>
              <p>OCR results can contain mistakes. Review extracted text before relying on it.</p>
            </details>
            <span>© {new Date().getFullYear()} OCRify</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default TextConverter;