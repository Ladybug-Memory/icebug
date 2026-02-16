# Development Practices

## Git Workflow

1. **Create feature branch:** `git checkout -b feature/my-feature`
2. **Build and test locally** before submitting PR
3. **Run relevant tests** to verify changes don't break existing functionality
4. **Follow coding style** guidelines in AGENTS.md
5. **Rebase on development branch** before creating PR (prefer linear history)

## PR Guidelines

- PRs require at least one review before merging
- PRs should be open for 7 days after last substantial change
- Critical bugfixes are exempt from the 7-day policy
- Use force push to update PRs on your fork

## Breaking Changes

- Breaking API changes must be documented in CHANGES.md
- Include migration guidelines for users
- Breaking ABI changes are allowed on major releases
- Minor/patch releases guarantee ABI stability

## Deprecation Policy

- Features need to be deprecated for 2 releases before removal
- Use deprecation warnings in C++ code

## Compiler Support

- Support all releases of GCC and Clang for 5 years
- Use C++11 features only (more widely compatible)

## Code Review

- All PRs need at least one review from a maintainer
- Reviewer cannot be the PR author
- Address feedback before merging
