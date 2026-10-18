#include <stdio.h>
#include <string.h>

typedef int (*ArgMain)(int argc, const char *argv[]);
typedef int (*VoidMain)(void);

typedef struct {
    const char *name;
    const char *description;
    ArgMain argMain;
    VoidMain voidMain;
    int expensive;
} Command;

#define ARG_COMMAND(name, description, symbol, expensive) \
    {name, description, symbol, NULL, expensive}
#define VOID_COMMAND(name, description, symbol, expensive) \
    {name, description, NULL, symbol, expensive}

#if defined(ANE_FIELDFARE_PRIVATE_SUITE)
int ANEFieldfareCompressedConstantMain(void);
int ANEFieldfareCompressionSpellingMain(void);
int ANEFieldfareNativeDtypeProbeMain(void);
int ANEFieldfareNativeDtypeMain(void);
int ANEFieldfareProgramResidencyMain(int, const char *[]);
int ANEFieldfareResidentProjectionMain(int, const char *[]);
int ANEFieldfarePrivateSurfaceMain(void);
int ANEFieldfareSystemCompressedModelMain(int, const char *[]);
int ANEFieldfareCompressedComputePlanMain(int, const char *[]);
int ANEFieldfareCoreMLCompileMain(int, const char *[]);
int ANEFieldfareAOTCompressedProjectionMain(int, const char *[]);
int ANEFieldfareAOTResidencyContentionMain(int, const char *[]);

static const Command commands[] = {
    VOID_COMMAND("compressed-constant", "private in-memory compressed-constant constructions",
                 ANEFieldfareCompressedConstantMain, 1),
    VOID_COMMAND("compression-spelling", "compiler spelling matrix for compressed constants",
                 ANEFieldfareCompressionSpellingMain, 1),
    VOID_COMMAND("native-dtype-probe", "native dtype compiler acceptance probe",
                 ANEFieldfareNativeDtypeProbeMain, 1),
    VOID_COMMAND("native-dtype", "native dtype numerical gate",
                 ANEFieldfareNativeDtypeMain, 1),
    ARG_COMMAND("program-residency", "device-global program residency lifecycle",
                ANEFieldfareProgramResidencyMain, 1),
    ARG_COMMAND("resident-projection", "resident projection numerical gate",
                ANEFieldfareResidentProjectionMain, 1),
    VOID_COMMAND("private-surface", "private ANE selector inventory",
                 ANEFieldfarePrivateSurfaceMain, 0),
    ARG_COMMAND("system-compressed-model", "Apple-shipped compressed artifact archaeology",
                ANEFieldfareSystemCompressedModelMain, 0),
    ARG_COMMAND("compressed-compute-plan", "Core ML compressed-model compute plan",
                ANEFieldfareCompressedComputePlanMain, 0),
    ARG_COMMAND("coreml-compile", "Core ML package compilation artifact gate",
                ANEFieldfareCoreMLCompileMain, 1),
    ARG_COMMAND("aot-compressed-projection", "AOT compressed projection oracle",
                ANEFieldfareAOTCompressedProjectionMain, 1),
    ARG_COMMAND("aot-residency-contention", "AOT residency ceiling and contention lifecycle",
                ANEFieldfareAOTResidencyContentionMain, 1),
};
static const char *suiteName = "private";
#elif defined(ANE_FIELDFARE_TRANSPORT_SUITE)
int ANEFieldfareDequantMain(void);
int ANEFieldfareCarrierWriteMain(void);
int ANEFieldfarePrepBreakdownMain(int, const char *[]);
int ANEFieldfarePositionLaneMain(int, const char *[]);
int ANEFieldfareTransportCeilingMain(int, const char *[]);
int ANEFieldfareExpertLayoutMain(int, const char *[]);
int ANEFieldfareExplicitReadMain(int, const char *[]);
int ANEFieldfareExpertResidencyMain(int, const char *[]);
int ANEFieldfareSustainedRampMain(int, const char *[]);

static const Command commands[] = {
    VOID_COMMAND("dequant", "bit-exact compact dequantization oracle",
                 ANEFieldfareDequantMain, 0),
    VOID_COMMAND("carrier-write", "carrier expansion microbenchmark and oracle",
                 ANEFieldfareCarrierWriteMain, 0),
    ARG_COMMAND("prep-breakdown", "production expert preparation attribution",
                ANEFieldfarePrepBreakdownMain, 0),
    ARG_COMMAND("position-lane", "multi-position ANE expert dispatch oracle",
                ANEFieldfarePositionLaneMain, 0),
    ARG_COMMAND("transport-ceiling", "cold expert-aligned storage transport matrix",
                ANEFieldfareTransportCeilingMain, 1),
    ARG_COMMAND("expert-layout", "packed expert access-order and layout gate",
                ANEFieldfareExpertLayoutMain, 0),
    ARG_COMMAND("explicit-read", "explicit-read generation comparison",
                ANEFieldfareExplicitReadMain, 1),
    ARG_COMMAND("expert-residency", "physical cold-versus-warm mmap expert control",
                ANEFieldfareExpertResidencyMain, 1),
    ARG_COMMAND("sustained-ramp", "hot expert host-versus-ANE sustained timing",
                ANEFieldfareSustainedRampMain, 0),
};
static const char *suiteName = "transport";
#else
#error "select one ANE Fieldfare benchmark suite"
#endif

static const size_t commandCount = sizeof(commands) / sizeof(commands[0]);

static void printRegistry(FILE *stream, const char *program) {
    fprintf(stream, "usage: %s <subcommand> [arguments...]\n", program);
    fprintf(stream, "%s suite subcommands:\n", suiteName);
    for (size_t i = 0; i < commandCount; ++i)
        fprintf(stream, "  %-28s %s%s\n", commands[i].name, commands[i].description,
                commands[i].expensive ? " [opt-in]" : "");
}

int main(int argc, const char *argv[]) {
    if (argc == 2 && (!strcmp(argv[1], "--list") || !strcmp(argv[1], "list"))) {
        for (size_t i = 0; i < commandCount; ++i)
            printf("%s\t%s\n", commands[i].name,
                   commands[i].expensive ? "opt-in" : "safe");
        return 0;
    }
    if (argc < 2 || !strcmp(argv[1], "--help") || !strcmp(argv[1], "help")) {
        printRegistry(argc < 1 ? stderr : stdout, argc < 1 ? "benchmark-suite" : argv[0]);
        return argc < 2 ? 2 : 0;
    }
    for (size_t i = 0; i < commandCount; ++i) {
        if (strcmp(argv[1], commands[i].name)) continue;
        return commands[i].argMain ? commands[i].argMain(argc - 1, argv + 1)
                                   : commands[i].voidMain();
    }
    fprintf(stderr, "unknown %s suite subcommand: %s\n", suiteName, argv[1]);
    printRegistry(stderr, argv[0]);
    return 2;
}
