#import <Foundation/Foundation.h>

#include "Tokenizer.h"

#include <stdlib.h>
#include <string.h>

@interface ANEFieldfareTokenizerObject : NSObject
@property(nonatomic, readonly) NSDictionary<NSString *, NSNumber *> *vocabulary;
@property(nonatomic, readonly) NSArray<NSString *> *tokens;
@property(nonatomic, readonly) NSDictionary<NSString *, NSNumber *> *mergeRanks;
@property(nonatomic, readonly) NSDictionary<NSData *, NSNumber *> *exactIgnorableMergeRanks;
@property(nonatomic, readonly) NSDictionary<NSString *, NSNumber *> *specialTokens;
@property(nonatomic, readonly) NSSet<NSNumber *> *specialIDs;
- (instancetype)initWithPath:(NSString *)path;
@end

static NSString *pairKey(NSString *left, NSString *right) {
    return [NSString stringWithFormat:@"%@\uFFFF%@", left, right];
}

static NSData *pairByteKey(NSString *left, NSString *right) {
    NSData *leftBytes = [left dataUsingEncoding:NSUTF8StringEncoding];
    NSData *rightBytes = [right dataUsingEncoding:NSUTF8StringEncoding];
    uint64_t leftLength = leftBytes.length;
    NSMutableData *key = [NSMutableData dataWithCapacity:sizeof(leftLength) + leftBytes.length + rightBytes.length];
    [key appendBytes:&leftLength length:sizeof(leftLength)];
    [key appendData:leftBytes];
    [key appendData:rightBytes];
    return key;
}

static bool containsBOM(NSString *string) {
    const char *bytes = string.UTF8String;
    return bytes && strstr(bytes, "\xEF\xBB\xBF") != NULL;
}

@implementation ANEFieldfareTokenizerObject
- (instancetype)initWithPath:(NSString *)path {
    self = [super init];
    if (!self) return nil;
    NSData *data = [NSData dataWithContentsOfFile:path options:NSDataReadingMappedIfSafe error:nil];
    if (!data) return nil;
    NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    NSDictionary *model = [root isKindOfClass:NSDictionary.class] ? root[@"model"] : nil;
    NSDictionary *vocab = [model[@"vocab"] isKindOfClass:NSDictionary.class] ? model[@"vocab"] : nil;
    NSArray *merges = [model[@"merges"] isKindOfClass:NSArray.class] ? model[@"merges"] : nil;
    if (!vocab || !merges || vocab.count == 0 || vocab.count > UINT32_MAX) return nil;

    NSUInteger vocabularySize = 0;
    for (NSNumber *identifier in vocab.allValues)
        vocabularySize = MAX(vocabularySize, identifier.unsignedIntegerValue + 1);
    if (!vocabularySize || vocabularySize > UINT32_MAX) return nil;
    NSMutableArray *idTokens = [NSMutableArray arrayWithCapacity:vocabularySize];
    for (NSUInteger i = 0; i < vocabularySize; ++i) [idTokens addObject:[NSNull null]];
    for (NSString *token in vocab) {
        NSNumber *identifier = vocab[token];
        if (![token isKindOfClass:NSString.class] || identifier.unsignedLongLongValue >= vocabularySize) return nil;
        idTokens[identifier.unsignedIntegerValue] = token;
    }
    /* NSString deliberately compares U+FEFF as ignorable, but tokenizer BPE
     * keys are byte-exact. Restore the three pairs Foundation coalesces. */
    NSDictionary<NSNumber *, NSString *> *ignorablePairs = @{
        @715: @"//", @135260: @"\uFEFF//",
        @8510: @"<?", @140291: @"\uFEFF<?",
        @208867: @"\uFEFF#", @236865: @"#",
    };
    for (NSNumber *identifier in ignorablePairs)
        idTokens[identifier.unsignedIntegerValue] = ignorablePairs[identifier];

    NSMutableDictionary<NSString *, NSNumber *> *ranks = [NSMutableDictionary dictionaryWithCapacity:merges.count];
    NSMutableDictionary<NSData *, NSNumber *> *exactIgnorableRanks = [NSMutableDictionary dictionaryWithCapacity:8];
    [merges enumerateObjectsUsingBlock:^(id raw, NSUInteger rank, BOOL *stop) {
        (void)stop;
        if (![raw isKindOfClass:NSArray.class] || [raw count] != 2) return;
        NSString *left = raw[0], *right = raw[1];
        if ([left isKindOfClass:NSString.class] && [right isKindOfClass:NSString.class]) {
            ranks[pairKey(left, right)] = @(rank);
            if (containsBOM(left) || containsBOM(right)) exactIgnorableRanks[pairByteKey(left, right)] = @(rank);
        }
    }];
    /* Foundation also treats U+FEFF as ignorable while keying merge pairs.
     * Retain the seven byte-distinct entries from the installed merge table. */
    NSArray<NSArray *> *bomMerges = @[
        @[@"\u2581", @"\uFEFF", @103314], @[@"\uFEFF", @"\uFEFF", @218439],
        @[@"\uFEFF", @"//", @316134], @[@"\uFEFF", @"<?", @326606],
        @[@".", @"\uFEFF", @352371], @[@"\uFEFF\uFEFF", @"\uFEFF\uFEFF", @407578],
        @[@"\uFEFF", @"#", @461747],
    ];
    for (NSArray *merge in bomMerges) exactIgnorableRanks[pairByteKey(merge[0], merge[1])] = merge[2];

    NSMutableDictionary *special = [NSMutableDictionary dictionary];
    NSMutableSet *specialIDs = [NSMutableSet set];
    for (NSDictionary *entry in root[@"added_tokens"]) {
        if (![entry isKindOfClass:NSDictionary.class] || ![entry[@"special"] boolValue]) continue;
        NSString *content = entry[@"content"];
        NSNumber *identifier = entry[@"id"];
        if ([content isKindOfClass:NSString.class] && [identifier isKindOfClass:NSNumber.class]) {
            special[content] = identifier;
            [specialIDs addObject:identifier];
        }
    }
    _vocabulary = [vocab copy];
    _tokens = [idTokens copy];
    _mergeRanks = [ranks copy];
    _exactIgnorableMergeRanks = [exactIgnorableRanks copy];
    _specialTokens = [special copy];
    _specialIDs = [specialIDs copy];
    return self;
}
@end

struct ANEFieldfareTokenizer { void *object; };

static ANEFieldfareTokenizerObject *object(ANEFieldfareTokenizer *tokenizer) {
    return tokenizer ? (__bridge ANEFieldfareTokenizerObject *)tokenizer->object : nil;
}

static NSNumber *vocabularyID(ANEFieldfareTokenizerObject *tokenizer, NSString *piece) {
    const char *bytes = piece.UTF8String;
    if (bytes) {
        if (strcmp(bytes, "//") == 0) return @715;
        if (strcmp(bytes, "\xEF\xBB\xBF//") == 0) return @135260;
        if (strcmp(bytes, "<?") == 0) return @8510;
        if (strcmp(bytes, "\xEF\xBB\xBF<?") == 0) return @140291;
        if (strcmp(bytes, "\xEF\xBB\xBF#") == 0) return @208867;
        if (strcmp(bytes, "#") == 0) return @236865;
    }
    return tokenizer.vocabulary[piece];
}

ANEFieldfareTokenizer *ANEFieldfareTokenizerOpen(const char *path) {
    if (!path) return NULL;
    @autoreleasepool {
        NSString *string = [NSString stringWithUTF8String:path];
        ANEFieldfareTokenizerObject *value = string ? [[ANEFieldfareTokenizerObject alloc] initWithPath:string] : nil;
        if (!value) return NULL;
        ANEFieldfareTokenizer *tokenizer = calloc(1, sizeof(*tokenizer));
        if (!tokenizer) return NULL;
        tokenizer->object = (__bridge_retained void *)value;
        return tokenizer;
    }
}

void ANEFieldfareTokenizerClose(ANEFieldfareTokenizer *tokenizer) {
    if (!tokenizer) return;
    if (tokenizer->object) CFRelease(tokenizer->object);
    free(tokenizer);
}

static void appendOrdinary(ANEFieldfareTokenizerObject *tokenizer,
                           NSString *text,
                           NSMutableArray<NSNumber *> *output) {
    if (!text.length) return;
    text = [text stringByReplacingOccurrencesOfString:@" " withString:@"\u2581"];
    NSMutableArray<NSString *> *pieces = [NSMutableArray array];
    /* Hugging Face BPE starts from Unicode scalar values, not extended
     * grapheme clusters. Keep a UTF-16 surrogate pair together, but leave
     * combining scalars separate so merge ranks remain authoritative. */
    for (NSUInteger cursor = 0; cursor < text.length;) {
        unichar first = [text characterAtIndex:cursor];
        NSUInteger length = 1;
        if (CFStringIsSurrogateHighCharacter(first) && cursor + 1 < text.length &&
            CFStringIsSurrogateLowCharacter([text characterAtIndex:cursor + 1])) length = 2;
        [pieces addObject:[text substringWithRange:NSMakeRange(cursor, length)]];
        cursor += length;
    }

    while (pieces.count > 1) {
        NSUInteger bestPosition = NSNotFound, bestRank = NSUIntegerMax;
        for (NSUInteger i = 0; i + 1 < pieces.count; ++i) {
            NSString *left = pieces[i], *right = pieces[i + 1];
            NSNumber *rank = (containsBOM(left) || containsBOM(right))
                ? tokenizer.exactIgnorableMergeRanks[pairByteKey(left, right)]
                : tokenizer.mergeRanks[pairKey(left, right)];
            if (rank && rank.unsignedIntegerValue < bestRank) {
                bestRank = rank.unsignedIntegerValue;
                bestPosition = i;
            }
        }
        if (bestPosition == NSNotFound) break;
        pieces[bestPosition] = [pieces[bestPosition] stringByAppendingString:pieces[bestPosition + 1]];
        [pieces removeObjectAtIndex:bestPosition + 1];
    }

    for (NSString *piece in pieces) {
        NSNumber *identifier = vocabularyID(tokenizer, piece);
        if (identifier) {
            [output addObject:identifier];
            continue;
        }
        NSData *bytes = [piece dataUsingEncoding:NSUTF8StringEncoding];
        const uint8_t *raw = bytes.bytes;
        for (NSUInteger i = 0; i < bytes.length; ++i) {
            NSNumber *byteID = vocabularyID(tokenizer, [NSString stringWithFormat:@"<0x%02X>", raw[i]]);
            [output addObject:byteID ?: @3];
        }
    }
}

bool ANEFieldfareTokenizerEncode(ANEFieldfareTokenizer *rawTokenizer,
                                 const char *utf8,
                                 ANEFieldfareTokenIDs *result) {
    if (!result) return false;
    *result = (ANEFieldfareTokenIDs){0};
    ANEFieldfareTokenizerObject *tokenizer = object(rawTokenizer);
    if (!tokenizer || !utf8) return false;
    @autoreleasepool {
        NSString *text = [NSString stringWithUTF8String:utf8];
        if (!text) return false;
        NSArray<NSString *> *specials = [tokenizer.specialTokens.allKeys
            sortedArrayUsingComparator:^NSComparisonResult(NSString *left, NSString *right) {
                if (left.length > right.length) return NSOrderedAscending;
                if (left.length < right.length) return NSOrderedDescending;
                return [left compare:right];
            }];
        NSMutableArray<NSNumber *> *ids = [NSMutableArray array];
        NSUInteger cursor = 0, ordinaryStart = 0;
        while (cursor < text.length) {
            NSString *matched = nil;
            for (NSString *candidate in specials) {
                if (candidate.length <= text.length - cursor &&
                    [text compare:candidate options:0 range:NSMakeRange(cursor, candidate.length)] == NSOrderedSame) {
                    matched = candidate;
                    break;
                }
            }
            if (!matched) { ++cursor; continue; }
            appendOrdinary(tokenizer, [text substringWithRange:NSMakeRange(ordinaryStart, cursor - ordinaryStart)], ids);
            [ids addObject:tokenizer.specialTokens[matched]];
            cursor += matched.length;
            ordinaryStart = cursor;
        }
        appendOrdinary(tokenizer, [text substringFromIndex:ordinaryStart], ids);
        uint32_t *values = ids.count ? malloc(ids.count * sizeof(*values)) : NULL;
        if (ids.count && !values) return false;
        for (NSUInteger i = 0; i < ids.count; ++i) values[i] = ids[i].unsignedIntValue;
        *result = (ANEFieldfareTokenIDs){values, ids.count};
        return true;
    }
}

void ANEFieldfareTokenIDsFree(ANEFieldfareTokenIDs *tokens) {
    if (!tokens) return;
    free(tokens->values);
    *tokens = (ANEFieldfareTokenIDs){0};
}

char *ANEFieldfareGemmaUserPrompt(const char *utf8UserText) {
    if (!utf8UserText) return NULL;
    @autoreleasepool {
        NSString *user = [NSString stringWithUTF8String:utf8UserText];
        if (!user) return NULL;
        user = [user stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
        NSString *prompt = [NSString stringWithFormat:
            @"<bos><|turn>user\n%@<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", user];
        return strdup(prompt.UTF8String);
    }
}

char *ANEFieldfareTokenizerDecode(ANEFieldfareTokenizer *rawTokenizer,
                                  const uint32_t *tokens,
                                  size_t count,
                                  bool skipSpecialTokens) {
    ANEFieldfareTokenizerObject *tokenizer = object(rawTokenizer);
    if (!tokenizer || (count && !tokens)) return NULL;
    @autoreleasepool {
        NSMutableString *result = [NSMutableString string];
        NSMutableData *byteRun = [NSMutableData data];
        void (^flushBytes)(void) = ^{
            if (!byteRun.length) return;
            NSString *decoded = [[NSString alloc] initWithData:byteRun encoding:NSUTF8StringEncoding];
            if (decoded) [result appendString:decoded];
            else {
                const uint8_t *bytes = byteRun.bytes;
                for (NSUInteger i = 0; i < byteRun.length; ++i) [result appendFormat:@"\uFFFD"];
                (void)bytes;
            }
            [byteRun setLength:0];
        };
        NSRegularExpression *bytePattern = [NSRegularExpression regularExpressionWithPattern:@"^<0x([0-9A-Fa-f]{2})>$" options:0 error:nil];
        for (size_t i = 0; i < count; ++i) {
            if (tokens[i] >= tokenizer.tokens.count) { flushBytes(); [result appendString:@"<unk>"]; continue; }
            NSNumber *identifier = @(tokens[i]);
            if (skipSpecialTokens && [tokenizer.specialIDs containsObject:identifier]) { flushBytes(); continue; }
            id stored = tokenizer.tokens[tokens[i]];
            if (![stored isKindOfClass:NSString.class]) { flushBytes(); [result appendString:@"<unk>"]; continue; }
            NSString *piece = stored;
            NSTextCheckingResult *match = [bytePattern firstMatchInString:piece options:0 range:NSMakeRange(0, piece.length)];
            if (match) {
                NSString *hex = [piece substringWithRange:[match rangeAtIndex:1]];
                unsigned value = 0;
                [[NSScanner scannerWithString:hex] scanHexInt:&value];
                uint8_t byte = (uint8_t)value;
                [byteRun appendBytes:&byte length:1];
            } else {
                flushBytes();
                [result appendString:[piece stringByReplacingOccurrencesOfString:@"\u2581" withString:@" "]];
            }
        }
        flushBytes();
        return strdup(result.UTF8String);
    }
}

uint32_t ANEFieldfareTokenizerVocabularySize(ANEFieldfareTokenizer *tokenizer) {
    NSUInteger count = object(tokenizer).tokens.count;
    return count <= UINT32_MAX ? (uint32_t)count : 0;
}
