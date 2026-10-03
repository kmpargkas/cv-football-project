# Running this on your own footage

Point the pipeline at a clip of yours and it produces `demo.mp4`: the footage with
each player marked at their feet in their kit colour and numbered, the ball ringed
where it was detected, and a minimap of everyone's position on the pitch.

## What the footage needs

- **A tactical camera:** high, wide, showing most of the pitch. The detector is
  fine-tuned for this view; broadcast footage with close-ups and replays will not work.
- **No hard cuts.** Calibration follows the camera between labelled frames, and a cut
  breaks that.
- **Visible pitch markings.** You calibrate by clicking line crossings, so a clip
  where the lines cannot be seen cannot be calibrated.
- **High resolution, ideally 2.5K or more.** The ball is only 10–15 px across at
  2940 px wide. Lower resolution shrinks it further and the ball is found less often.

## Setup, once

```bash
uv sync
uv run python scripts/pull_weights.py   # the two detection models
```

## The three steps

```bash
# 1. Probe the clip and write its config. Prints the frames to label next.
uv run python scripts/new_clip.py --video data/raw/newmatch.mp4

# 2. Label pitch points on those frames (~10 minutes of clicking).
uv run python scripts/annotate_pitch_points.py --config configs/newmatch.yaml \
    --frames 0,99,198,297,... --output data/annotations/pitch_points_newmatch.json

# 3. Run everything.
uv run python scripts/run_pipeline.py --config configs/newmatch.yaml
```

Step 1 prints the exact commands for steps 2 and 3 with your clip's names and frame
list already filled in, so you can copy them from there.


Step 3 takes about 20 minutes for a 15-second clip, almost all of it in the `players`
stage. The result is `outputs/newmatch/demo.mp4`.

## Labelling pitch points: the one step quality depends on

The annotator shows each suggested frame and asks for named line crossings in turn.
Click to place one, `s` to skip one you cannot see, `u` to go back one, `n` for the
next frame, `q` to save and quit.

- **Label every frame the tool suggests.** The spacing comes from how far calibration
  can be carried between labelled frames before it drifts. On a 1390-frame clip,
  8 labelled frames gave 0.94 m error and 15 passed comfortably.
- **Aim for 14 or more points per frame, including some on the far half.** Points
  are the cheapest way to improve accuracy: going from 8 to 10 points on one frame
  halved the error of the frames calibrated from it (1.72 m → 0.88 m).

### The calibration check

The pipeline's `gate` stage holds out each labelled frame in turn, rebuilds that
frame's calibration from the others, and measures the error against the points you
clicked. If the median is over 0.5 m it stops the run, because everything downstream
would inherit the error. The check is deliberately pessimistic: a held-out frame is
as far from labelled help as any frame gets. It never trusts an earlier result: it
runs again every time.

When it fails, open `outputs/newmatch/holdout_metrics.json` and look at the failing
entries' `seeded_from` field. That names the labelled frame whose points produced
the bad calibration, which is usually not the frame being scored. Add points to that
frame, then run step 3 again.

## Re-running

Each stage is skipped when its outputs already exist, so a re-run after a crash picks
up where it stopped. To redo a stage and everything after it:

```bash
uv run python scripts/run_pipeline.py --config configs/newmatch.yaml --from players
uv run python scripts/run_pipeline.py --config configs/newmatch.yaml --dry-run   # show the plan only
```

The stages are `calibrate`, `gate`, `players`, `stitch`, `ball`, `project`, `radar`
and `annotate`. After relabelling points, use `--from calibrate`. To redraw only the
video, use `--from annotate`, which takes seconds.

## What you get

All in `outputs/newmatch/`:

| file | what it is |
|---|---|
| `demo.mp4` | the deliverable: footage with players, ball and minimap drawn on it |
| `radar.mp4` | the top-down view on its own, with a running possession split |
| `positions.csv`, `ball_positions.csv` | per-frame positions in pitch metres |
| `possession.csv` | per-frame possession: `team_a`, `team_b`, `keeper` or `unknown` |
| `teams.csv`, `palette.json` | each track's role and team, and the measured kit colours |
| `holdout_metrics.json` | the calibration check's result |

## Optional settings

**Team names.** `team_a` is always the darker kit (more precisely, whichever comes
first on the colour channel where the two kits differ most), and the `players` stage
prints which one it picked:

```
TeamID: team_a is the darker kit (L 139 vs 234, the widest Lab channel)
```

To label them in the possession summary, uncomment the `match:` block at the end of
the config and fill in `team_a_name` / `team_b_name`. They are display names only.

**Pitch size.** Set `pitch.length` and `pitch.width` from the venue if you know them;
the default is 105 × 68 m. Wrong values do not raise an error: they silently scale
every distance, speed and possession radius.

**No player numbers.** To render without them, run `scripts/render_tracking.py` with
`--no-ids`.

## When it will not work well

- **Two similar kits.** If the two teams' kits cannot be told apart, the fit refuses
  rather than guessing, and the `players` stage stops with an error.
- **Kits close to grey.** A team in a washed-out kit is drawn in a near-neutral colour
  that is hard to tell apart from the grey used for unclassified players.
- **Fluorescent yellow-green referee kits.** These can be mistaken for grass when
  sampling kit colour, which leaves the referee unclassified.
- **A low camera.** The minimap sits over the bottom of the frame and can cover the
  near touchline.
- **Lost tracks.** When the tracker loses a player and picks them up again, they come
  back with a new number.
