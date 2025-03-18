# Space-Rats
An AI bot is tasked with capturing a Space Rat in the ship. The bot consistently updates it's knowledge base of the rat using Markov Prediction Model.

# Abstract
This project involves designing a customized autonomous bot to navigate a structured, grid-based
environment representing the layout of a ”ship” and locate a target, the ”space rat.” The environment
is modeled as a 30x30 grid, where approximately 60% of the cells are open, and the outer edge cells are
permanently blocked. The bot faces two main challenges: it has no knowledge of its initial location due
to memory loss from cosmic interference, and it must track down the space rat within the grid.
The bot operates in two key phases: Localization and Target Pursuit. In the Localization phase, the
bot iteratively determines its exact position by sensing the number of adjacent blocked cells and attempting
directional movements. By combining sensory data and movement outcomes, the bot systematically
narrows down potential locations until it identifies its precise p osition. In the Target Pursuit phase, the
bot uses a probabilistic detector to track the space rat. This detector ”pings” based on proximity, with
higher probability as the bot moves closer. If the bot and space rat occupy the same cell, the detector
confirms their alignment.
Our bot’s performance is compared against a baseline bot through various tests, focusing on metrics
such as the number of moves taken, sensing actions, and detector activations. To optimize tracking
e iciency, we analyze the sensitivity parameter- α, which affects the detector’s probability model. As α
increases, the likelihood of distant pings decreases, challenging tracking but preserving precision when
close; conversely, low α values lead to frequent pings, which may saturate the sensor. The analysis is plotted
as a function of α, providing insights into optimal settings.
In advanced tests, the space rat moves randomly after each bot action, introducing dynamic motion.
This requires an update to the bot’s knowledge base using adaptive probability calculations, enabling it
to anticipate and track the moving target. The comparative performance of the customized bot versus
the baseline in this dynamic environment highlights the effectiveness of adaptive tracking strategies.
