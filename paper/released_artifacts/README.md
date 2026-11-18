# Released artifacts and rule control (Section 6.4)

This folder reproduces Figure 6, the numbers of Section 6.4, and the appendix paragraphs on the released-artifact
rules and on rule replay. The artifacts are a transcoder on Qwen3-4B and a Gemma Scope SAE on Gemma-3-4B, each with
six rules: four categories and two kinds of copyrighted text.

## Numbers from results (CPU, seconds)

`results.json` holds every number, as `analyze.py` computed it from the paper's runs.

```bash
python paper/show.py released_artifacts           # every number of Section 6.4
python paper/show.py released_artifacts figure6   # Figure 6 alone
```

## Results from runs (CPU, minutes)

```bash
python paper/released_artifacts/analyze.py   # reads $RUNS/released_artifacts and writes results.json
```

`analyze.py` replays the rules on the final feature maxima of each Disabled run. A Disabled run of a policy
therefore shows what any subset of its rules would have refused, and `analyze.py` checks this replay against the
deployed policies request by request.

## Runs (GPU)

Prepare the inputs first (`../data/README.md`). The runs use the generation settings of
`../effectiveness/settings.sh`. The times are the generation time of the paper's runs, without model loading and
judging.

| Command | Runs | Time |
|---|---|---|
| `paper/released_artifacts/run.sh transcoder_matched` | the transcoder policy under matched settings | 0.2 h |
| `paper/released_artifacts/run.sh transcoder_defaults` | the transcoder policy with vLLM's default optimizations | 0.2 h |
| `paper/released_artifacts/run.sh sae_defaults` | the SAE policy and its Disabled arm with vLLM's default optimizations | 0.4 h |
| `paper/released_artifacts/run.sh combined` | probe rules and transcoder rules as one policy, and its Disabled arm | 1.2 h |
| `HARMBENCH=/path/to/HarmBench paper/released_artifacts/run.sh copyright` | the copyright requests | 0.6 h |

The transcoder replays on the combined policy's Disabled run, which records the transcoder's features under matched
settings. The SAE replays on its own Disabled run. The layout is `$RUNS/released_artifacts/EXPERIMENT/BENCHMARK/seed_S/ARM`,
and the copyright runs are under `copyright/{transcoder,sae}/{requests,xstest}/seed_S`.

## Where each number comes from

| In the paper | Key in `results.json` |
|---|---|
| Figure 6 | `figure6.{transcoder,sae}.RULE.COLUMN` |
| seven of the eight category rules fire most often on their own target | `text.category_rules_firing_most_on_their_target` |
| the records name a rule for each refusal and for no released response | `agreement_with_deployed_policies.{transcoder,sae}` |
| replaying all six rules agrees with the deployed policies to within a point | `all_rules`, `text.largest_gap_replay_to_deployed` |
| each transcoder category rule halves the ASR on its category, FRR at most 3.7% | `rules_alone.transcoder`, `text.transcoder_category_rules_*` |
| the transcoder's cybercrime rule, from 46.5% to 1.0% at an FRR of 2.7% | `text.transcoder_cybercrime_rule` |
| the SAE's chemical and biological rule, from 46.3% to 25.9%, 0.9 points of FRR | `text.sae_chemical_rule` |
| the SAE's cybercrime and misinformation rules fire on 9% of XSTest | `text.sae_xstest_firing` |
| each copyright rule fires on 92 to 96% of its own requests, at most 2% of the rest | `copyright`, `text.copyright_own_rule_fires`, `text.copyright_other_rule_fires_at_most` |
| the two copyright rules as a policy refuse 94 to 97% | `text.copyright_policy_refuses` |
| three copyright rules add at most 0.4 points of FRR, the transcoder's book rule 7 | `text.copyright_rules_added_frr` |
| 100 XSTest prompts and 156 HarmBench prompts were open to the rule search | `calibration_overlap` |
| without them, firing changes by at most 6 points on the target and 2 on XSTest | `held_out_firing_change` |
| held out, each transcoder category rule still halves its ASR, FRR at most 3.1% | `rules_alone_held_out.transcoder`, `text.held_out_transcoder_*` |
| held out, the SAE's chemical and biological rule, from 48.8% to 29.3%, 1.6 points | `text.held_out_sae_chemical_rule` |
| 37 copyright requests for books and 37 for songs | `copyright.held_out_requests` |
| the copyright classifier marks 1.4% and 4.1% of the answers | `copyright.{transcoder,sae}.reproduced_by_disabled_answers` |
| each copyright policy's XSTest FRR is within 0.3 points of the replay | `text.copyright_policy_frr_minus_replayed` |
| the replay agrees with each of the 2,556 requests of each deployed policy | `agreement_with_deployed_policies.{transcoder,sae}` |
| the replay contains the first rule of each of the 1,806 Disabled requests | `agreement_with_deployed_policies.transcoder_disabled_first_rule` |
