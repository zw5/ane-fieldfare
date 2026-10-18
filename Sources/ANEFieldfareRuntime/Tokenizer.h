#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ANEFieldfareTokenizer ANEFieldfareTokenizer;

typedef struct {
    uint32_t *values;
    size_t count;
} ANEFieldfareTokenIDs;

/* Loads the installed tokenizer.json. The returned object is immutable and
 * may be shared by prompt and decode code. */
ANEFieldfareTokenizer *ANEFieldfareTokenizerOpen(const char *tokenizerJSONPath);
void ANEFieldfareTokenizerClose(ANEFieldfareTokenizer *tokenizer);

/* Tokenizes text exactly as supplied. Special tokens registered by the
 * tokenizer remain atomic. The caller owns result.values. */
bool ANEFieldfareTokenizerEncode(ANEFieldfareTokenizer *tokenizer,
                                 const char *utf8,
                                 ANEFieldfareTokenIDs *result);
void ANEFieldfareTokenIDsFree(ANEFieldfareTokenIDs *tokens);

/* The no-tools, no-system Gemma 4 instruction template installed alongside
 * this tokenizer, with add_generation_prompt=true and thinking disabled. */
char *ANEFieldfareGemmaUserPrompt(const char *utf8UserText);

/* Returns a malloc-owned UTF-8 string. Byte-fallback runs are reconstructed
 * before UTF-8 decoding. Set skipSpecialTokens for user-visible generation. */
char *ANEFieldfareTokenizerDecode(ANEFieldfareTokenizer *tokenizer,
                                  const uint32_t *tokens,
                                  size_t count,
                                  bool skipSpecialTokens);

uint32_t ANEFieldfareTokenizerVocabularySize(ANEFieldfareTokenizer *tokenizer);

#ifdef __cplusplus
}
#endif
