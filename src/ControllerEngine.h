#if defined(_WIN32)
#define CE_EXPORT __declspec(dllexport)
#else
#define CE_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

CE_EXPORT int RunControllerEngine(void);

#ifdef __cplusplus
}
#endif