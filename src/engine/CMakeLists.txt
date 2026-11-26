willm_add_module(engine
    SOURCES
        options.cpp
    DEPS
        willm::model
        willm::kv_cache
        willm::sampler
        willm::tokenizer)
