# 01 — Rebuild the GND vocabulary from the official release

**What to build:**

The vocabulary files currently on disk are a lossy derivative: some earlier, unreproducible step
stripped GND's comma-qualifiers from preferred names, synonyms and related terms, altering 18,043 of
79,427 tib-core entries. Those qualifiers are GND's homograph disambiguators, so stripping them merges
distinct senses of the same word into one string — exactly the ambiguity the label tower has to resolve.

Regenerate both vocabulary files faithfully from the shared-task release, and settle how a qualified
term is rendered as text. The chosen rendering turns the raw comma form into a parenthetical, because
that reads as natural language to an encoder while preserving the sense distinction:

    raw       Interaktion,Naturwissenschaft
    rendered  Interaktion (Naturwissenschaft)

Rendering is a flag on the label-text builder, defaulting to the parenthetical form, so raw and
stripped forms remain available as ablations rather than being baked in.

After this ticket the whole dataset — records and vocabulary — is reproducible from the official
repository by one command, with nothing large tracked in version control.

**Blocked by:** None — can start immediately

**Status:** done

- [x] Both vocabulary files regenerate from a sparse clone of the official repository with no manual steps
- [x] Regenerated files preserve every qualifier present in the official release
- [x] Qualifier rendering is a documented flag with three modes: parenthetical (default), raw, stripped
- [x] A count of affected entries and terms is recorded so the effect of each mode is measurable
- [x] The rebuild command is documented in the README and produces both records and vocabulary
- [x] Nothing in the regenerated dataset is tracked by git
