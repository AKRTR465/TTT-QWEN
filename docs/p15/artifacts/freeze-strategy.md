# P15 freeze strategy

Synthetic/tiny engineering gate only. Qwen is fully frozen with an empty Qwen allowlist. Static Fast Adapter W0 and the explicit state modules are the only Outer AdamW candidates. Predictor, functional SGD, transient W_t, Bank/FSM runtime state, caches, and full-model weights are excluded. This is not a convergence or scientific-benefit claim.
