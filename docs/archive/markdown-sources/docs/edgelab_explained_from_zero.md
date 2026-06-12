# The Edge Lab, Explained From Zero

You do not need any computer background to understand this. I am going to build it up piece by piece, using a single story that runs through the whole thing. Read it slowly, in order. By the end you will understand every machine, every program, and how they all fit together.

---

## Part 1: The one idea behind the whole lab

Imagine you are a student with a stack of math problems to solve. You have two ways to get each problem done:

**Option 1: Solve it yourself at your desk.**
You are not super fast at math, so each problem takes you a while. But you are always available, and there is no waiting. You just do it.

**Option 2: Mail the problem to your genius friend across town.**
Your friend solves any math problem almost instantly. But you have to mail the problem to them and wait for the answer to come back. Normally the mail is quick. But sometimes the roads are jammed and the mail takes forever. And sometimes your friend is busy solving other people's problems, so yours sits in a pile waiting its turn.

So here is the question the whole lab is about:

**For each problem, is it better to solve it yourself (slow but reliable), or mail it to your genius friend (usually faster, but risky when the roads are bad or your friend is swamped)?**

That is the entire lab. Everything else is just the machinery that makes this question real and measurable.

Now let me translate the story into the real words:

- "Solve it yourself at your desk" is called **local processing** or **edge computing**. The desk is the Raspberry Pi.
- "Mail it to your genius friend" is called **remote processing**. The friend is the GPU server.
- "The roads" are the **network**.
- "Your friend being busy" is **GPU load**.
- The skill of deciding, problem by problem, which option to pick is what the students have to figure out.

Hold onto this story. I will keep pointing back to it.

---

## Part 2: What is the actual "problem" being solved?

In our lab, the math problems are video frames, and the task is finding a tennis ball.

There is a video of a tennis ball moving around. A video is just a flip-book: a long series of still pictures shown quickly one after another. Each still picture is called a **frame**.

For every frame, we want to answer one question: **where is the tennis ball in this picture?** The answer is a pair of numbers, an x and a y, which is just the position (like "across 300, down 150").

Finding the ball is done by an **AI model**. An AI model is a program that has been trained to recognize things in pictures. Ours is called **YOLOv10n**. You hand it a picture, it hands back the position of the ball. You do not need to know how it works inside. Think of it as a very specialized pair of eyes that only knows how to spot a ball and say where it is.

So now the story is exact:
- A "problem" = one video frame.
- "Solving the problem" = running YOLOv10n on that frame to find the ball.
- "Solve it yourself" = run YOLOv10n on the Raspberry Pi.
- "Mail it to your friend" = send the frame to the GPU server, which runs YOLOv10n and mails back the position.

---

## Part 3: Why is this a real tradeoff? Why not always use the fast friend?

If the genius friend is always faster, why would anyone ever solve it themselves? Because of two things that can go wrong with mailing it away.

**Thing 1: The roads can get bad (network delay).**
Normally the mail is fast. But we can deliberately jam the roads. When we do, your frame takes a long time to reach your friend, and the answer takes a long time to come back. Suddenly the "fast friend" is slow, because most of the time is spent traveling, not solving.

**Thing 2: The friend can get swamped (GPU load).**
We can also flood your friend with a pile of other people's problems. Now even though your friend solves each problem instantly, your problem has to wait in line behind all the others. So again, the "fast friend" becomes slow.

This is why it is a genuine decision. When the roads are clear and your friend is free, mailing it away wins easily. When the roads are jammed or your friend is buried, you are better off just doing it yourself at your desk, slowly but without the waiting.

The students cannot see into the future. They have to watch the current conditions (how bad are the roads right now? how busy is the friend right now?) and make a smart call each moment. That is the skill the lab is teaching.

---

## Part 4: How do we measure who did a good job? (the clever part)

Here is the neat trick that turns this into a game with a score.

The ball is moving. By the time you get an answer about where the ball is, a little time has passed, and the ball has already moved on. So your answer is always a little bit out of date. It tells you where the ball *was*, not where it *is right now*.

The longer you took to get the answer, the more the ball has moved, and the more wrong your answer is.

We show this on screen with two dots:
- A **green dot**: where the ball truly is right now. We know this for certain because we figured out the true position of the ball in every frame ahead of time, on a powerful computer, with no rush. This pre-computed answer key is called the **ground truth**.
- A **red dot**: where your answer says the ball is. This is your guess, and it arrives a little late.

If you were instant, the red dot would sit right on top of the green dot. The slower you are, the further the red dot lags behind the green dot, because the ball kept moving while you waited.

The distance between the red dot and the green dot is called the **displacement**. Small displacement means you were fast and accurate. Big displacement means you were slow.

We add up the displacement over the whole experiment. That total is your **score**. Lower is better. The student group with the lowest total wins.

So the chain is: faster answers, less time for the ball to move, red dot closer to green dot, smaller displacement, better score. Speed is everything, and speed depends on making the right local-versus-remote choice as conditions change.

---

## Part 5: The cast of characters (the machines)

There are four kinds of computers in this lab. Let me introduce each one with its job in the story, then its real name and address.

### The Raspberry Pi (your desk)

A Raspberry Pi is a tiny, cheap, weak computer, about the size of a deck of cards. There are four of them, one for each student group. The Pi is "you at your desk": the local worker. It is slow, but it is right there and always available.

The Pi does several jobs at once:
- It plays the video and looks at each frame.
- It can run YOLOv10n itself (slowly) to find the ball. This is the "solve it yourself" option.
- It can mail a frame to the genius friend instead. This is the "send it away" option.
- It runs the student's decision-making brain that chooses between those two.
- It draws the green and red dots on screen and keeps the score.

The Pi sits at home with the student. It reaches the other machines over the internet through a secure tunnel called a **VPN** (think of the VPN as a special pass that lets the Pi into the university's private network from outside).

### The GPU Server (your genius friend), at address 172.22.174.145

This is a powerful computer with a special chip called a **GPU** (an NVIDIA H100, which is extremely fast and expensive). The GPU is brilliant at running AI models quickly. This machine is the "genius friend across town."

It does not just sit there waiting, though. We need a program that takes incoming frames, runs the model, and sends back answers. That program is called **Triton**. Triton is like the friend's front desk: you mail your frame to Triton, Triton runs YOLOv10n on the GPU, and Triton mails back the ball position.

This machine also runs a couple of helper programs we will cover in Part 6.

### The Network VM (the stretch of road we can sabotage), at address 172.22.174.148

"VM" stands for **virtual machine**, which is just a computer that exists as software rather than a physical box. For your purposes, treat it as a normal computer.

This machine sits in the middle of the road between the Pi and the GPU server. Every frame the Pi mails to the friend passes through this machine first, and every answer coming back passes through it too. It is a toll booth on the highway.

Why have a toll booth in the middle? Because this is where we can deliberately make the road bad. This machine has a built-in Linux tool called **tc** (short for "traffic control") that can add delay, add jitter (random wobble in the delay), or drop mail entirely. When we want the "network load" part of the experiment, we tell this machine to slow everything passing through it. That is how we sabotage the roads on demand.

This is also exactly why the Pi must mail its frames *through* this machine and not straight to the friend. The slowdown only happens at the toll booth. If the Pi took a different road that skipped the toll booth, the slowdown would never apply, and that whole part of the lab would do nothing.

### The SeQaM Machine (the game master and the scoreboard), at address 172.22.174.149

This machine wears three hats.

**Hat 1: the game master, called SeQaM.** Somebody has to decide when the roads get bad and when the friend gets swamped, on a schedule, so the experiment is fair and repeatable. SeQaM is a program that follows a script: at 0 seconds do nothing, at 30 seconds make the friend busy, at 60 seconds jam the roads, and so on. It carries out these changes by reaching into the other machines and running commands on them.

**Hat 2: the message board, called Kafka.** All these machines need to tell each other what is happening. "The roads are now jammed." "The friend is now busy." "We are now in phase 2." Kafka is a shared bulletin board where any machine can pin up a note, and any machine can read the notes it cares about. More on this in Part 7, because it is important.

**Hat 3: the big screen, called Grafana.** Grafana takes all the notes from the message board and draws them as live graphs: a line showing how busy the friend is, a line showing how jammed the roads are, a line showing your latency. This is how students *see* the conditions instead of staring at raw numbers. It is the scoreboard and status display.

---

## Part 6: What program runs on each machine (the full list)

Now that you know the machines, here is exactly what is running on each one and why.

### On the Raspberry Pi
- **Your app** (the `client` folder, started by `main.py`). This is the heart of everything the student touches. I break it down fully in Part 8.
- **A local copy of the YOLOv10n model** (the file `yolov10n.onnx`). This is so the Pi can run the model itself for the "solve it yourself" option.
- **The video file and the ground truth file** (the answer key). Needed to play frames and to score.

### On the GPU Server (172.22.174.145)
- **Triton**: the front desk that runs the model on the GPU and answers incoming frame requests.
- **The GPU metrics publisher**: a small program that constantly checks "how busy is the GPU right now?" and pins that number to the Kafka message board. This is how students can see whether the friend is swamped.
- **The GPU load script** (`load.sh`): a program that, when run, floods Triton with a pile of fake requests to deliberately make the friend busy. The game master runs this during the "GPU load" phase.

### On the Network VM (172.22.174.148)
- **The traffic control script** (`tc_control.sh`): the tool that jams the road (adds delay and jitter) when the game master tells it to.
- **The network conditions publisher**: a small program that constantly reports "how jammed are the roads right now?" (current delay and jitter) to the Kafka message board, so students can see it.
- **The phase publisher**: a small program that watches for the game master writing the current phase name into a file, and announces that phase to the message board so the Pi knows which phase is happening.
- **Forwarding rules**: the setting that makes this machine pass traffic through to the GPU server (the toll booth letting cars continue down the highway).

### On the SeQaM Machine (172.22.174.149)
- **SeQaM**: the game master following its timed script.
- **Kafka**: the message board.
- **Grafana**: the big screen with the graphs.

---

## Part 7: The message board (Kafka) and how machines talk

This is worth its own section because it is how the whole system stays coordinated.

Picture a giant bulletin board in the middle of a room. The board has several labeled sections, and anyone can pin a note in a section, and anyone can stand and watch a section to read new notes as they appear. In the real system, the board is **Kafka**, and the labeled sections are called **topics**.

Here are the sections of our board:

- **Section "/edgelab/server/metrics"**: the GPU server pins notes here saying how busy it is. ("GPU is 80% busy. 95 requests per second.")
- **Section "/edgelab/network/metrics"**: the network VM pins notes here saying how bad the roads are. ("Delay is 100 milliseconds. Jitter is 20.")
- **Section "/edgelab/server/events/phase"**: the current phase name gets pinned here. ("We are now in phase: network_load.")
- **Section "/edgelab/app/metrics/group1"**: the Pi pins its own results here. ("Frame 412, I went remote, latency was 130ms, displacement was 18 pixels.")

Why does this matter? Because the student's decision-making brain needs to *see the conditions* to make smart choices. It stands and watches the first three sections of the board. When it sees "roads are jammed" and "friend is busy", it can decide "I should solve this one myself." That is the entire point of the board: it lets the brain react to what is happening.

Grafana, the big screen, also watches the board and turns every section into a live graph.

---

## Part 8: Your app, explained piece by piece

Your app runs on the Pi. Inside, it is split into four workers that run at the same time, each doing one job, passing work to the next. In computing, these simultaneous workers are called **threads**. Think of them as four people on an assembly line.

Here are the four workers and what each does:

### Worker 1: the Frame Reader
Its only job is to open the video and pull out frames one at a time, in order, and hand each frame to the next worker. Like someone flipping through the flip-book and handing each page down the line.

### Worker 2: the Dispatcher
This worker takes a frame and asks the big question: **local or remote?** It does not decide that itself. It asks the student's brain (Worker 4) for the current decision. Then:
- If the decision is "local", it runs YOLOv10n on the Pi itself.
- If the decision is "remote", it mails the frame through the toll booth to Triton on the GPU server and waits for the answer.
- If it tries remote and the mail fails (roads totally broken), it falls back to doing it locally so the system never gets stuck. This is called **fallback**.

Either way, it ends up with a ball position (the red dot) and hands it to the next worker.

### Worker 3: the Scorer
This worker compares the answer to the ground truth (the answer key). It works out the green dot (true position) and the red dot (your answer), measures the distance between them (the displacement), and adds it to the running score. It also draws both dots on the video frame so you can watch it live, and it pins the result to the Kafka message board.

There is one special case worth knowing: sometimes the model finds no ball at all (maybe the frame was sent late and was skipped, or the ball was hard to see). When that happens, instead of scoring it as wildly wrong, the app adds a fixed, fair penalty. This keeps a single dropped frame from wrecking the whole score and hiding the real story, which is about speed.

### Worker 4: the SP-Agent (the student's brain)
**SP** stands for "Service Placement", which is a fancy way of saying "deciding where to place the work" (local or remote). This is the one piece the students actually write.

This worker stands and watches the message board (the GPU busyness, the road conditions, the current phase). Based on what it sees, it repeatedly decides: should we be doing frames locally or remotely right now? It writes that decision somewhere the Dispatcher can read it.

The student's whole task is to write the logic inside one function called `decide()`, which returns either the word `"local"` or the word `"remote"`. A simple version might say: "if the roads are jammed OR the friend is busy, do it myself; otherwise mail it away." A smarter version might track how things have been going over the last few seconds. That is the creative part of the lab, and where groups compete.

The students only edit one small file (`sp_agent.py`). Everything else (the four-worker pipeline, the scoring, the message board connections) is already built for them. They just supply the brain.

---

## Part 9: The experiment phases (the schedule of sabotage)

The game master (SeQaM) runs a fixed, repeating cycle so every group faces the same conditions. The cycle is 120 seconds long, split into four phases of 30 seconds each, then it loops forever.

1. **baseline (0 to 30 sec)**: nothing is sabotaged. Roads clear, friend free. Mailing it away should win easily here.
2. **gpu_load (30 to 60 sec)**: the friend gets swamped with fake work. The roads are still clear, but the friend is slow to get to your problem. Doing it yourself might now be competitive.
3. **network_load (60 to 90 sec)**: the friend is free again, but now the roads are jammed. Mailing it away is slow because of travel time. Doing it yourself probably wins.
4. **combined (90 to 120 sec)**: both at once. Friend swamped and roads jammed. Doing it yourself almost certainly wins.

The game master announces each phase change by writing the phase name into a file on the network VM, which the phase publisher then pins to the message board, which the Pi reads. That is how your app always knows which phase it is in.

Because the conditions keep changing, a brain that always picks "remote" will do badly during the bad phases, and a brain that always picks "local" will do badly during the good phases. The winning brain is the one that switches at the right moments. That is the lesson.

---

## Part 10: Following one frame all the way through (the grand tour)

Let me tie everything together by following a single frame from start to finish, naming every part as it goes.

1. The **Frame Reader** on the Pi pulls frame number 500 out of the video and hands it down the line.

2. The **Dispatcher** on the Pi asks the **SP-Agent** (the student's brain): "local or remote?" The brain has been watching the **Kafka message board**. Right now it sees the roads are clear and the friend is free (we are in the **baseline** phase), so it says "remote."

3. The Dispatcher mails frame 500 toward the **GPU server**. The frame travels through the **Network VM** (the toll booth). Since this is the baseline phase, the toll booth is not slowing anything down, so the frame passes straight through.

4. The frame arrives at **Triton** on the GPU server. Triton runs **YOLOv10n** on the **GPU**, finds the ball, and gets a position. It mails that position back, again through the toll booth, back to the Pi.

5. The Dispatcher now has the ball position (the **red dot**) and hands it to the **Scorer**.

6. The Scorer looks up the **ground truth** for frame 500 (the **green dot**, the true position). It measures the distance between red and green (the **displacement**). Because the roads were clear and the friend was fast, the answer came back quickly, the ball barely moved, and the red dot is very close to the green dot. Small displacement. Good.

7. The Scorer adds this small displacement to the running **score**, draws both dots on the frame so the student can watch, and pins a note to the Pi's section of the **message board**: "frame 500, remote, fast, small displacement."

8. **Grafana** (the big screen) sees that note and updates its graphs.

Now imagine the same frame during the **network_load** phase instead. At step 3, the toll booth deliberately holds the frame for an extra 100 milliseconds each way. The answer comes back late. By then the ball has moved a lot, so the red dot lags far behind the green dot. Big displacement. Bad. If the student's brain had noticed the jammed roads and chosen "local" instead, it would have skipped the toll booth entirely, done the work on the Pi, and gotten a fresher (if slightly slower to compute) answer. That is the judgment call the whole lab trains.

---

## Part 11: The shortest possible summary

- The lab teaches one question: do the work locally (slow but reliable) or send it to a fast remote server (faster, unless the network or the server is overloaded)?
- The work is finding a tennis ball in each video frame using an AI model.
- The Raspberry Pi is the local worker. The GPU server is the fast remote worker. The network VM is the road in between, which we can sabotage. The SeQaM machine runs the schedule, the message board, and the graphs.
- Latency makes answers arrive late, so the ball has moved by the time you know where it was. The distance between the true position and your late answer is your score. Lower is better.
- Students write one small piece of logic that watches the conditions and decides, moment by moment, whether to go local or remote. Choosing well across changing conditions is the entire skill.

Whenever you get lost in the technical roadmap, come back to the homework story: you, your genius friend, the roads, and your friend's pile of other work. Every machine and every program is just one part of that story made real.
