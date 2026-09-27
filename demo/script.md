# Scrubline demo script (2:30)

**0:00** Space Coast pads have shown they can fly 143 launches a year. They fly 73. Weather is the biggest cause of scrubs we can name: 91 of 154, and it's growing every year.
**0:15** Each weather scrub costs about 670 thousand dollars, the attempt plus a day of delay. That's about 61 million since 2021, and moving just the riskiest 10 percent of hours avoids almost half.
**0:30** Scrubline is a launch weather officer: a weather model trained on every past attempt, all ten lightning rules with real values, a Qwen model fine-tuned on River, and GBrain memory.
**0:45** Let's ask about a bad hour: Falcon 9 from SLC-40, October 1st at 8 AM. Qwen fans out the tools in parallel and searches GBrain memory itself.
**1:00** The call is NO-GO. The simulation holds the countdown and scrubs on rain, with the forecast cloud layers and freezing level drawn in.
**1:12** Every launch rule shows a value against its limit: lightning distance, electric field, cloud-top temperature.
**1:24** Because this hour is bad, it suggests better windows, each with a full rule check. Let's take the first one.
**1:36** A day later the rules clear and the weather risk is low. Both models, LightGBM and our River-tuned Qwen, agree.
**1:51** So watch it fly: liftoff, max-Q, stage separation, orbit.
**2:06** The planner colors every hour of the week by the call, so a team picks its window before it commits the pad.
**2:18** Weather scrubs are preventable. Scrubline makes the call, with receipts, and learns from every correction.
