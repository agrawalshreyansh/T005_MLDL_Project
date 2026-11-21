# Presenter guide for the slides

This guide lets anyone present the 16-slide deck, even with no background in the project.
Read sections 1–3 once (10 minutes), then follow section 4 while you present.

Files: `deep_learning/presentation/index.html` (open in Chrome, animated) or
`deep_learning/presentation/presentation.pdf` (same slides, still images).
Talk length: **about 12 minutes** (a 5-minute version is in section 7).

---

## 1. The project in 60 seconds

Los Angeles has 207 speed sensors on its freeways. Every 5 minutes each one records the average
speed of the cars passing it. We want to predict each sensor's speed 5, 15, 30 and 60 minutes ahead.

Traffic jams **spread along the road**: when cars stop at one sensor, the queue grows backwards to
the sensors behind it. So a sensor's future depends on its own recent past **and** on what its
neighbours on the road are doing right now.

We compared two kinds of models:

* **Models that look at each sensor alone** (simple rules, XGBoost, a plain neural network).
* **Graph models** that also look at the road network: which sensor feeds into which
  (ST-GAT and DCRNN).

**Result:** the graph model ST-GAT is the most accurate an hour ahead, 5% better than the best
model without learned neighbour use and 8.6% better in rush hour. When we take the road connections
away from ST-GAT and change nothing else, the advantage disappears. So the improvement comes
from the road network, not just from using a neural network.

---

## 2. Words you need (plain meanings)

| Word | Meaning in one line |
|---|---|
| **Sensor / detector** | a loop in the road that measures how fast cars go; there are 207 |
| **METR-LA** | the public dataset: 207 LA sensors, every 5 minutes, March–June 2012 (119 days) |
| **Horizon** | how far ahead we predict: 5, 15, 30 or 60 minutes |
| **MAE** | mean absolute error: on average, how many mph the prediction is off. **Lower is better** |
| **Persistence** | the simplest guess: "speed in an hour = speed now". The baseline to beat |
| **XGBoost** | a strong classic machine-learning model (many decision trees); our Phase 1 model |
| **XGBoost + neighbours** | the same, also given nearby sensors' speeds as extra numbers |
| **Neural network** | a model that learns patterns from data by adjusting many weights |
| **GRU** | a neural network that reads a sequence (the last hour) step by step |
| **Graph** | the road network as dots (sensors) and arrows (road connections) |
| **Upstream / downstream** | upstream = where traffic comes from; downstream = where it goes |
| **ST-GAT** | our graph model: GRU + "attention", each sensor learns which neighbours to listen to |
| **Attention** | learned weights that say how much each neighbour matters at this moment |
| **DCRNN** | a known graph model from 2018: information spreads along roads like a random walk |
| **No-edge twin / ablation** | ST-GAT with all road connections removed, everything else identical. It tests whether the roads cause the gain |
| **Train / validation / test** | days used to learn / to tune settings / to give the final score (never seen before) |
| **Seed** | a random starting point for training; we used 3 to check results are not luck |
| **Bootstrap / 95% CI** | a way to put error bars on a result by resampling test days; if the bar does not cross 0, the difference is real |
| **Peak** | rush hour: weekdays 07–10 and 15–19 |
| **Mask** | marking missing readings so the model ignores them |

---

## 3. Before you present

* **HTML version:** open `index.html` in Chrome, press **F** for full screen.
  **→ / Space** next slide, **←** back, **N** shows the speaker notes at the bottom (only you should
  see them on a second screen; otherwise leave them off).
* **PDF version:** open `presentation.pdf` in any PDF viewer, full-screen mode. No animation,
  everything else is identical.
* The map in the background is **real data** from one test day, Friday 15 June 2012.
  Dots = sensors, lines = road connections, red = slow, green = fast, grey rings = sensor offline.
* Bottom-left clock (on map slides) = the time of day shown. Bottom-right green sign = slide number.

---

## 4. Slide by slide

Each slide: **what is on screen**, **what to say** (you can read it almost word for word),
and the **bridge** to the next slide.

### Slide 1 — "Gridlock, forecast" (≈40 s)
**On screen:** title, the question, five team names; behind it the LA freeway map with red (slow) roads.
**Say:** "We are Team T005. Our question: can 207 freeway sensors in Los Angeles predict their own
traffic jams better if they listen to their neighbours? Behind this slide is a real day from our
test data, Friday 15 June 2012. Each dot is a sensor, each line a road connection, red means slow."
**Bridge:** "Why would neighbours help? Because traffic is contagious."

### Slide 2 — "Traffic is contagious" (≈50 s)
**On screen:** short explanation; red jams on the map.
**Say:** "A crash on one freeway doesn't stay there. The queue grows backwards, sensor by sensor.
So a sensor's future speed depends on its own past *and* on what is already happening upstream
and downstream. Our task: from the last 60 minutes at every sensor, predict the speed 5, 15, 30
and 60 minutes ahead."
**Bridge:** "Before modelling, we had to deal with a problem in the data."

### Slide 3 — "The data has holes" (≈50 s)
**On screen:** two big numbers, 8.1% and 2,148; grey rings on the map.
**Say:** "8.1% of all readings are missing. The dataset stores them as 0 mph. In 2,148 five-minute
moments every single sensor reads zero at once, which is impossible in real traffic: those are
outages. If we treated zeros as real, the model would learn fake traffic jams. So we mark them as
missing: the model is told a reading is missing, and missing values are never used to score it."
**Bridge:** "We are not starting from zero: Phase 1 already set a benchmark."

### Slide 4 — "Phase 1 set the bar" (≈40 s)
**On screen:** three sentences about XGBoost.
**Say:** "In Phase 1 we built XGBoost, a strong classic model. It beat the simple 'speed stays the
same' guess by 31% one hour ahead. Giving it the neighbours' speeds as extra inputs helped
further, which tells us the road network carries useful information. But XGBoost only sees
fixed summaries we designed by hand; it cannot learn *how* a jam travels."
**Bridge:** "To learn that, the model needs the road network itself."

### Slide 5 — "The road is a graph, and it has direction" (≈50 s)
**On screen:** one highlighted sensor with its connections, yellow-orange and blue lines.
**Say:** "We describe the roads as a graph: 1,515 one-way connections between sensors, built from
real driving distances, not straight lines. Only 27% go both ways, because a freeway has a
direction. Upstream sensors (yellow-orange) feed traffic into this sensor; downstream sensors
(blue) are where its queue will spread. We keep that direction in every model."
**Bridge:** "We then built models step by step, changing one thing at a time."

### Slide 6 — "One idea per step" (≈60 s) — *the most important plan slide*
**On screen:** four boxes: 1, 1b, 2, 3.
**Say:** "Step 1 is a neural network that only looks at time: each sensor on its own.
Step 2, ST-GAT, adds the road graph: each sensor reads its neighbours and learns how much each
one matters. Step 3, DCRNN, is a well-known graph model from the research literature.
Step 1b is the key experiment: it is exactly ST-GAT, same size, same training, but with every
road connection removed. If 1b is as good as ST-GAT, the graph doesn't matter. If it is worse,
the graph is the reason."
**Bridge:** "How does ST-GAT decide which neighbours matter?"

### Slide 7 — "Attention: who should I listen to?" (≈50 s)
**On screen:** one sensor with lines to its neighbours (moving pulses in the HTML version).
**Say:** "First, a small network summarises each sensor's last hour. Then every sensor sends that
summary to its neighbours. A learned scorer gives each incoming message a weight; the weights
add up to one and change every moment. So at 8 am a sensor might listen mostly to the one just
upstream, at midnight mostly to itself. Upstream and downstream get separate attention."
**Bridge:** "A fair comparison needs fair training."

### Slide 8 — "Training, fairly" (≈50 s)
**On screen:** five bullet points.
**Say:** "All models use the same split by date: 83 days to learn, 12 to tune, and the last
24 days for the final test, cut at midnight so no day is shared. The models predict the
*change* from the last observed speed. Settings were chosen only on the 12 tuning days, for
ST-GAT and its no-edge twin separately, so tuning can't create the difference. We trained on a
free Kaggle GPU, and touched the test days only once, at the very end."
**Bridge:** "Here are the results."

### Slide 9 — "The graph models pull away as the horizon grows" (≈60 s)
**On screen:** line chart, error (y) against how far ahead (x). Lower is better.
**Say:** "Lower is better. At 5 minutes all learned models are about equal: the latest reading
already tells you most of it. The further ahead we predict, the more they separate. One hour
ahead: persistence is off by 5.35 mph, XGBoost with neighbours 3.65, the time-only network
3.63, DCRNN 3.49, and ST-GAT is best at 3.46 mph."
**Point at:** the green ST-GAT line at 60 min, lowest.
**Bridge:** "But is it really the graph? That is the next slide."

### Slide 10 — "Switch the graph off and the gain disappears" (≈60 s) — *the key result*
**On screen:** green and orange bars per horizon, with small error bars.
**Say:** "Green bars: how much better ST-GAT is than its twin with no road connections. It is
3 to 5% better at every horizon, and the error bars never touch zero, so it is not luck.
Training the models again with different random starts changes the result by only about
0.01 mph, ten times less than this effect. Orange bars: ST-GAT against XGBoost with
neighbours, up to 5.3% better one hour ahead. And the twin without roads is barely better than
the plain time-only network. Conclusion: the road graph is what helps."
**Bridge:** "Where does that matter most? In rush hour."

### Slide 11 — "Rush hour is where it counts" (≈45 s)
**On screen:** blue (off-peak) and red (peak) bars.
**Say:** "Rush hour here means weekdays 7–10 am and 3–7 pm, fixed before training. One hour
ahead, ST-GAT is 8.6% better than XGBoost with neighbours in rush hour, but 3.4% off-peak.
The graph helps most exactly when traffic changes fastest, which is when forecasts are most
useful."
**Bridge:** "And where on the map does it help?"

### Slide 12 — "Where the graph helps" (≈40 s)
**On screen:** map with green (graph helps) and red (graph hurts) dots.
**Say:** "Each dot compares ST-GAT with its no-road twin at that sensor. Green: the graph helps,
at 76% of sensors. Red: it is slightly worse, but by at most 0.17 mph, while gains reach
0.92 mph. And it is not simply the sensors with the most connections that gain."
**Bridge:** "Let's see the forecast itself."

### Slide 13 — "The forecast, replayed" (≈40 s)
**On screen:** the same map as slide 1, but now showing ST-GAT's prediction made 30 minutes earlier.
**Say:** "This is the same day as the title slide, but every dot now shows what ST-GAT predicted
half an hour before. Over the whole day it is off by 3.6 mph on average. It follows the evening
slowdown, but sudden drops still arrive late: no model here can predict a crash before it happens."
**Bridge:** "How do our numbers compare with published research?"

### Slide 14 — "Against the literature" (≈40 s)
**On screen:** table of published models and ours (orange rows).
**Say:** "Top rows are results published in research papers on this same dataset; orange rows are
ours. Our simple seasonal average matches the published one, 4.18 vs 4.16, and our graph models
are in the same range as the published graph models. The setups differ slightly, so this is a
sanity check, not a competition."
**Bridge:** "Finally, what we do *not* claim."

### Slide 15 — "What we do not claim" (≈40 s)
**On screen:** four bullet points.
**Say:** "One city, 119 days in 2012, speed only: we claim nothing beyond that. Attention weights
describe where the model looks, not proof of cause. Our DCRNN is our own re-implementation.
And nothing here controls real traffic lights; the traffic-signal part of the project is a
simulation only."

### Slide 16 — "See it live" (≈20 s + demo)
**Say:** "Every forecast for every sensor and every test day is in our interactive dashboard.
Thank you, we're happy to take questions." If there is time, switch to the dashboard and follow
`deep_learning/dashboard/PRESENTER_GUIDE.md`.

---

## 5. Questions you may get (short answers)

| Question | Answer |
|---|---|
| Why not just use XGBoost? | It was good, but ST-GAT is 5.3% better one hour ahead and 8.6% in rush hour, and the ablation shows the gain comes from learning on the road graph. |
| How do you know the graph causes the improvement? | The no-edge twin is identical except for the road connections; it is 3–5% worse at every horizon and the error bars exclude zero. |
| Isn't it just luck? | Three random starts differ by about 0.01 mph; the effect is ten times bigger. Error bars come from resampling the 24 test days. |
| Why split by date and not randomly? | Random splits put tomorrow's traffic in the training data, which is cheating. Real forecasting only knows the past. |
| What did you do with missing data? | Zeros are outages, not stopped cars. We flag them in the input and never score on them. |
| Why is DCRNN not the winner? | It is as good up to 30 minutes and slightly worse at 60; it is also about 3× slower to train. |
| What does the model actually take as input? | The last 60 minutes at all 207 sensors: speed, whether it was observed, and time of day / day of week. |
| How long did training take? | About 8 hours on a free Kaggle GPU (Tesla T4) for all runs. |
| Can this control traffic lights? | No. The project only forecasts. A separate simulation of one intersection exists but does not use these forecasts. |
| What would you do next? | Let the model learn the road graph itself (Graph WaveNet), more cities, more random starts. |
| Why do errors get bigger further ahead? | The future is less certain the further you look; the last reading tells you a lot about 5 minutes ahead and little about an hour ahead. |
| What is MAE 3.46 in plain words? | One hour ahead, the prediction is on average 3.46 mph off the real speed. |

---

## 6. Things not to say

* Don't say "we beat the state of the art". Say "we are in the same range as published models".
* Don't say a zero means "traffic stopped". It means "sensor offline".
* Don't say "attention shows why the model is right". It shows where it looks.
* Don't say the system controls or could control real traffic. It only forecasts.
* Don't round "5.3%" up to "much better". Use the numbers on the slides.

---

## 7. Short on time? The 5-minute version

Slides **1 → 2 → 6 → 9 → 10 → 11 → 15**, about 40 seconds each. Use the "Say" text above, and on
slide 6 spend the extra time on step 1b (the no-edge twin). That is the whole argument: the
question, the plan, the main result, the proof that it is the graph, rush hour, and the limits.
