"""Orchestration: the stage graph, the runner's shared guards, and new-clip onboarding.

Each stage is a separate script run in graph order; ``stages`` declares what each one
needs and produces, ``cli`` holds the checks those scripts share, and ``onboard``
scaffolds a config for a new clip.
"""
