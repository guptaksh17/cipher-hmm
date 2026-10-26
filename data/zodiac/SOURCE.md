# Zodiac 408 data sources

Downloaded 2026-10-26.

| File | Content |
|---|---|
| `z408_cipher.txt` | 408-symbol ASCII transcription of the Z408, 24 rows x 17 columns, 54 distinct symbols |
| `z408_solution.txt` | Known solution, one plaintext letter per cipher symbol, same layout |

## Sources

1. **Ciphertext and solution**: Dave Oranchak's `zodiac-killer-ciphers` project (the
   code behind zodiackillerciphers.com), file `Ciphers.txt`, constants
   `Z408_SOLUTION` and the `"Z408: Zodiac killer's solved 408 cipher"` entry:
   https://github.com/doranchak/zodiac-killer-ciphers/blob/master/src/main/java/com/zodiackillerciphers/ciphers/Ciphers.txt
2. **Cross-check of the ciphertext**: the same transcription as a 24 x 17 grid in
   https://github.com/doranchak/z408-symbol-position-stats/blob/HEAD/z408_symbol_locations.py
   (identical, character for character).

## Notes

* The transcription uses ASCII stand-ins for Zodiac's glyphs (for example `\`, `^`, `_`,
  `+`). The demo shows these characters as they are.
* The solution is the symbol-aligned decryption of the message as published, with
  the author's misspellings and enciphering slips kept (for example "forrest",
  "moat dangertue anamal", "paradice", "sloi down") and the unexplained final 18
  letters "ebeorietemethhpiti". Accuracy is evaluated against this text.
* Validation: every one of the 54 symbols decodes to exactly one letter in this
  solution, so it is a consistent homophonic key. Letters absent from the plaintext:
  j, q, z.
