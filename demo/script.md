# Scrubline demo script (~3:26)

**0:00** Space Coast launch pads have shown they can fly 143 times a year. They fly 73.
**0:12** When a countdown stops, weather is the biggest cause we can name: 91 of the 154 scrubs with a known cause. And it's growing, from 6 weather scrubs in 2021 to 27 in 2025.
**0:24** Each weather scrub costs the attempt, propellant, range and team, plus a day of delay. About 670 thousand dollars.
**0:37** That's roughly 61 million dollars lost to weather at the Cape since 2021. And our model says moving just the riskiest 10 percent of launch hours would have avoided almost half of it.
**0:50** So we built Scrubline, a launch weather officer. Every past attempt, a weather model trained on what forecasts actually said, and all ten lightning launch rules with real values.
**1:04** A Qwen model we fine-tuned on River gives a second opinion, a Qwen officer runs the whole flow, and GBrain holds the memory of every mission it learns from.
**1:18** Let's ask it about a bad hour: Falcon 9 from SLC-40, October 1st, 8 AM UTC. Qwen parses the question and fans out the tools in parallel.
**1:32** In the trace you can see the forecast, the rule estimates, LightGBM, our fine-tuned Qwen and GBrain memory all running at once, then Qwen searching GBrain itself before it writes the call.
**1:46** The call is NO-GO. The simulation holds the countdown and scrubs: rain on the flight path, with the forecast cloud layers and freezing level drawn in.
**2:00** Every launch rule shows a value against its limit: lightning distance, electric field in volts per meter, cloud-top temperatures. Estimates are labeled as estimates.
**2:14** Because this hour is bad, it suggests better windows nearby, each with a full rule check. Let's take the first one.
**2:28** Same site, a day later. Now the rules clear and the risk is low, so watch it fly: liftoff, max-Q, stage separation, orbit.
**2:44** Two independent models: LightGBM and our River-tuned Qwen. When both flag a launch, they're right four times out of five.
**2:58** And the planner shows every hour this week, colored by the call, so a team picks the window before it ever commits the pad.
**3:12** Weather scrubs are preventable. Scrubline turns 1,500 past misses into a call with receipts, and gets smarter every time the team corrects it.
