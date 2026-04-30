---
title: "Claude Code Meets Obsidian"
date: 2026-04-30
tags: ["ai", "obsidian", "claude-code", "adhd", "productivity"]
---

I have ADHD. My brain generates ideas faster than I can capture them, forgets what I said ten minutes ago, and would rather build an entire system from scratch than fill out a daily planner. I've tried every productivity tool, every note-taking app, every "second brain" framework. None of them stuck because they all required the one thing my brain won't do: consistent manual upkeep and structured input. They don't handle multi-modal input. They don't stitch ideas together automatically. They don't fill in the context gaps when I give half a thought and move on, because my brain already finished the sentence and jumped to the next thing. The assumption has always been that I'll remember the rest. I won't.

A boss told me once that if you're paying someone well into six figures, they shouldn't be buried in administrative overhead. There should be pooled assistants handling scheduling, expense reports, note-taking, so those people can spend their energy on the work they were actually hired to do. He was right. That world never materialized. But the need didn't go away.

So I stopped hoping for a personal assistant, stopped looking for the right tool, and started looking for a collaborator. One that could read every note I've ever written, remember what I said three weeks ago, turn a rambling voice memo into structured knowledge while I walk to the library (wind noise and all), and do the consistent upkeep my brain refuses to. This is how Claude Code became the operating system for my Obsidian vault, and why the combination finally works where everything else didn't.

## The Setup

Most people know Obsidian as a great note-taking app, and it is. But the thing that makes it work for this use case is simpler than that: it's markdown files on your filesystem. No proprietary database, no API needed to access your own data. Your notes aren't rows in someone else's database. They're files you own, in folders you control, on a disk you can touch. Just `.md` files. That matters because Claude Code operates on your filesystem. It reads files, writes files, searches across directories, and modifies content in place. Point it at an Obsidian vault and it can do everything you can do in the app, plus things the app was never designed to do.

The bridge between the two is a file called CLAUDE.md. It sits at the root of your vault and tells Claude how everything works: folder structure, naming conventions, tag hierarchy, daily note format, check-in and check-out flows, how to process voice memos, how to track people, how to manage tasks. It's not a prompt in the way most people think about prompts. It's closer to onboarding documentation for a new team member. "Here's how we do things. Read this before you touch anything."

Here's the part that surprises people: I haven't written a single line of it. Most engineers, especially those using Claude Code for actual code, spend significant time crafting and maintaining their CLAUDE.md files. I took a different approach. I let Claude write it. As I described what I wanted, as workflows evolved through trial and error, Claude updated the file itself. The conventions are more consistent than anything I would have written by hand, and they stay current because the thing maintaining them is the same thing following them. Every Monday during check-in, Claude reviews its own CLAUDE.md for drift, checking whether workflows have evolved through conversation but never been written back to the file. It's self-documenting in a way that no system I've built before has ever been. The best part though is the workflows in the file aren't a requirement, they're strong suggestions.

## The Daily Ritual: Check-In

For most of my career, the first thing I did every morning was open email and Slack. Check for fires. See what blew up overnight. The problem is that once you start chasing, you don't stop. One message leads to a thread, the thread leads to a meeting, the meeting generates three follow-ups, and by 2pm you've seemed productive all day without accomplishing a single damned thing you actually needed to do. Then you look back at the week and realize your task list didn't budge one inch. The fires were rarely fires. They were other people's "priorities" dressed up as urgency. Poor planning on someone else's part does not constitute an emergency on mine, but it took me years to actually believe that, and several more to actually put my own deliverables above someone else's poor planning.

The daily check-in breaks that pattern. Before I open anything else, I say "let's do a daily check-in" and Claude takes the lead. Here's what the CLAUDE.md tells it to do:

```markdown
### Daily Check-In Flow

Triggered when Wes says "let's do a daily check-in" (or similar).

1. **Preparation (silent)**: Check the actual system date. Read the previous 5 days of daily notes. Read the task list. Note any open items, patterns, follow-ups, or things that were mentioned but never resolved. If it's Monday, also generate the Weekly Brief.
2. **Context briefing**: Short rundown of what carried over, tasks approaching deadlines, anything he said he wanted to follow up on.
3. **Morning biomarkers** (ask all four together): Energy (1-10), Sleep (hours), Mental positivity (1-10), Physical well-being (1-10).
4. **Morning Planning questions** (one at a time, wait for response):
   - "What's the ONE thing that if you accomplished it today, it would make today a success?"
   - "Anything on your task list you're actively avoiding?"
   - "What stuck out from yesterday as something you'd like to improve on today?"
5. **Additional context questions**: Based on the 5-day review, ask 1-2 targeted follow-ups if relevant.
6. **Write to daily note**: Clean up responses, add inline tags, inject into the Morning Planning section.
```

That's it. That block of markdown drives a 10-minute conversation every morning that completely reframes how I start my day. Claude reads my recent history, tells me what I've been avoiding, what's overdue, and what I said I wanted to do. Then it asks me to commit to ONE thing. Not a list of ten priorities. One thing that would make today a success.

The biomarkers are tracked in the daily note's YAML front matter, which means they're queryable over time through Obsidian. I can look at my energy and mental state across weeks and see patterns I'd never notice in the moment. The morning I showed up at a 4 energy and a 6 mental after a weekend of landscaping, that's data. It tells me something about how physical exertion affects my work week.

The planning questions are asked one at a time, not dumped as a list. That's intentional. My brain will skip questions in a list. It won't skip a direct question from someone waiting for an answer. The "anything you're avoiding?" question has been particularly powerful. It surfaces procrastination patterns that my therapist and I are actively working on together, identifying the why behind the avoidance and how to move through it. Having that question asked every single morning means the pattern gets caught in days, not months.

> On days I have therapy, I'll have Claude produce a brief beforehand. It reviews recent daily notes, pulls out the themes, surfaces the patterns in what I've been avoiding or struggling with, and gives me something concrete to walk in with. The brief doesn't replace the session, but it means I show up prepared instead of spending the first twenty minutes trying to remember what happened since last time.

After I answer, Claude takes my spoken responses, cleans them up, adds tags for people and products I mentioned, and writes everything into the daily note in first person. I never open the file. I never format anything. I just talk.

On Mondays, the check-in also generates a weekly brief, which I'll cover in its own section below.

## The Daily Ritual: Check-Out

The check-out is the mirror. Before it even starts the conversation, Claude silently runs a vault backup and scans the inbox for anything that was dropped in during the day. Then it gives me a summary of what happened and we walk through the reflection.

```markdown
### Daily Check-Out Flow

Triggered when Wes says "let's check out" (or similar).

0. **Vault backup and voice memo check (silent)**: Run the backup script. Check for new voice memos.
1. **Inbox scan (silent)**: Scan _inbox/ for unprocessed items. Convert to markdown, add front matter and tags, file into the correct vault location.
2. **Day summary**: Review the daily note. Summarize what was accomplished, worked on, and created.
3. **Evening biomarkers** (all four together): Energy (1-10), Mental positivity (1-10), Physical well-being (1-10), Day rating (1-10).
4. **Afternoon Reflection questions** (one at a time):
   - "Did you complete the ONE thing?" (include what it was so I don't have to remember)
   - "How did the day feel overall?"
   - "Anything you're putting off or procrastinating on? Why?"
   - "What went well today?"
   - "What could be improved?"
   - "What do you want to tell tomorrow-Wes?"
   - "How do you want to show up for your family tonight?"
5. **Task extraction**: Scan the day's notes for task-shaped content. "I found these potential tasks — want me to add them?"
6. **Accomplishments extraction**: Draft accomplishments. Filter for real wins, not completed chores.
7. **Write to daily note**: Inject responses into Afternoon Reflection. Update Accomplishments and Linked Notes.
```

The reflection questions are where this stops being productivity theater and starts being genuinely useful. "Did you complete the ONE thing?" forces accountability. "Anything you're putting off?" surfaces avoidance patterns. "What do you want to tell tomorrow-Wes?" is the one that changed me. I'm writing a message to a version of myself who doesn't have today's context. It forces me to distill what actually matters. Sometimes it's words of affirmation, sometimes it's a Gibbs slap on the back of the head saying "get to work".

"How do you want to show up for your family tonight?" is the last question for a reason. It's the transition from work-brain to home-brain. Some days the answer is "present and engaged." Some days it's "I just need to be quiet tonight." Either way, I've thought about it intentionally instead of mindlessly heading downstairs still mentally at my desk.

The task extraction catches things I said during the day that are task-shaped but never made it to a list. Claude scans the day's notes and conversation history then presents them: "I found these potential tasks, want me to add them?" I confirm, it routes them to the right section with priority, time estimate, and a values-based "why." Tasks don't fall through cracks because they're extracted from conversation, not entered by discipline.

The accomplishments extraction is one of my favorite pieces. Claude drafts what it thinks my accomplishments were, but it filters. Completing a chore is not an accomplishment -- unless I've procrastinated on it for far too long. Processing inbox items in Obsidian is not an accomplishment. Deploying a cluster for the first time, building a mental model you can articulate, having a therapy breakthrough, finishing a deliverable, those are accomplishments. That filter came from a therapy session where I realized I was conflating "busy" with "productive." Collecting accomplishments also has a two-fold bonus: it makes annual reviews easier to prepare for, and it helps me show myself that I am actually moving forward and accomplishing things.

## The Voice Memo Pipeline

My best thinking doesn't happen at a desk. It happens while I'm walking, driving, or doing something with my hands. The problem has always been that those thoughts evaporate. I'd record a voice memo with the intention of going back to it later, and later never came. I have years of voice memos on my phone that I never listened to again. Hundreds of ideas, reflections, and plans that lived and died in an audio file.

The pipeline is simple. I record a voice memo on my phone using Apple Voice Memos. When I'm ready to process them, I tell Claude and it does the rest: converts the audio to WAV, runs it through whisper.cpp on my Mac's GPU, cleans up the transcript, and files the result into the vault.

The transcription itself is the least interesting part. What matters is the cleanup. Whisper outputs raw text with no punctuation, no formatting, and garbled proper nouns. "Harsh aura" becomes #harish-arora (the PM of the product I cover) with a link to his people file. "KVBM" (KV Cache Block Manager) doesn't get autocorrected to "cable management." Product names, people names, technical terms all get fixed because Claude has the full vault as context. It knows what I'm likely talking about because it knows what I've been working on.

Sometimes I talk directly to Claude in my memos. "Claude, file this under the Personal folder." "Strike that last part, I changed my mind." "This next section is a task, add it to the personal list." Claude listens. It parses those instructions out of the transcript and acts on them instead of including them in the output.

A 40-minute walking brain dump becomes a structured note with headers, wiki-links to related content, and proper tags in about five minutes of processing. I recorded one this morning about our outdoor remodel plans while walking to the library. Stream of consciousness about retaining walls, irrigation zones, porch designs, and city code research. Claude turned it into a structured project plan with sections, open questions, and a next-steps checklist. I didn't organize any of it. I just talked, often contradicting something I said earlier, Claude follows the train of thought and figures out the final meaning behind the rambling.

The voice memo pipeline is also how 76 old memos that had been sitting on my phone for months (and years) finally got processed. I asked Claude to work through them over the course of a few sessions. Personal reflections from years ago got filed into the right places. Work ideas got connected to current projects. Audio keepsakes of my son got moved to his folder. Things I'd said and forgotten became permanent, linked, searchable knowledge.

For my ADHD brain, this is the single most impactful workflow. The gap between having an idea and capturing it used to be "pull out my phone, open an app, type coherently, tag it, file it." Now it's "talk." That's it. The rest happens later, and it happens without me.

## The Inbox

The inbox is the simplest workflow and maybe the most satisfying. There's a folder in the vault called `_inbox/`. Anything I want processed goes there. PDFs, CSVs, voice memo transcripts, random files I downloaded. During check-out, Claude scans it silently. If something's there, it reads the content, converts it to proper markdown with front matter and tags, files it into the correct location in the vault, and mentions what it did during the day summary.

Today it processed a roof measurement PDF from an insurance report and a structural engineering report for a solar panel install. Both went from raw files in the inbox to properly named, tagged documents in `Personal/Home/` without me touching them. Yesterday it was voice memos. Last week it was meeting transcripts. The format doesn't matter. The destination doesn't matter. I drop it in, Claude figures out what it is and where it goes.

The principle is simple: reduce the friction between "I have information" and "it's in the right place with the right context" to zero. For someone whose brain generates tons of output constantly but has no patience to file it, that's not a convenience. It's the difference between my ideas being ephemeral, lost to time, and being able to reference and build upon them in the future.

## The Weekly Brief

Every Monday morning, before the check-in conversation starts, Claude generates a weekly brief. It reads all my daily notes from the previous week, reviews my task list, checks the training schedule I'm working through, and runs a handful of web searches for news related to what I'm actively working on. The output is a single document I reference all week.

The brief has five sections: what happened last week (pulled from daily notes, not invented), open items and carry-overs, the training plan for the week ahead, recommendations on where to focus my energy, and curated external news. The news section is the part that surprises people. It's not a generic industry roundup. Claude knows what I'm working on because it's been in every conversation I've had for weeks. So when it searches for news, it filters for things that connect to active projects and explains why each item matters in context.

> You might have noticed that there's a training plan for the week included in the weekly brief. Our industry is too fast paced and the company I work for, NVIDIA, is constantly innovating. My goal is to try and drip feed training to myself each week rather than letting months go by and realizing that everything I know about everything has changed.

Here's what the structure looks like (_note: the `Week nnn` title is a number counting from week one since I started at NVIDIA_):

```markdown
# Week nnn Brief — 2026-04-27 to 2026-05-01

## Previous Week Summary
  - Accomplishments (pulled from daily notes)
  - Key Meetings (attendees, decisions, action items)
  - Key Insight (the one thing that mattered most)
  - What Didn't Happen (accountability for what slipped)

## Open Items & Carry-Overs
  - Prioritized task list with due dates and status

## Training Plan
  - Day-by-day breakdown mapped to the training schedule

## Week Ahead Focus
  - Primary, secondary, and "consciously not doing this week"

## News & External Signals

### Google TurboQuant — KV Cache Compression to 3.5 Bits (ICLR 2026)
Google Research published TurboQuant at ICLR 2026, compressing KV cache
down to 3-4 bits per element with zero retraining — 8x inference speedup,
50%+ memory cost reduction. The technique uses PolarQuant (random rotation
matrix applied to K/V vectors before quantization) to redistribute variance uniformly...

**Why this matters for CMX:** TurboQuant directly compresses what CMX
stores. If KV cache per token shrinks 4-5x through quantization, the
capacity math for KVBM changes significantly — same rack stores 4-5x
more sessions...

### Prefill-as-a-Service — Cross-Datacenter KV Cache Transfer
...

### NVIDIA Leads $1B Round in VAST Data at $30B Valuation
...
```

Each news item isn't just a headline. Claude explains why it matters to what I'm actively working on. That connection is the difference between a news feed and intelligence.

The weekly brief is not a summary for a manager or a status report for a team. It's a briefing for future-me. Monday-me who hasn't thought about work since Friday. It's the document that says "here's where you left off, here's what's due, here's what changed in the world while you were away, and here's what I'd recommend you focus on." It's the briefing I wished someone would have written for me at every job I've ever had. Now something does.

## Task Management

I don't add tasks to a task list. I never have consistently, and I've accepted that I never will. Instead, tasks surface during conversation. I mention something during a check-in, complain about something during a check-out, or commit to a follow-up in the middle of a brain dump. Claude catches them.

At the end of each check-out, Claude scans everything from the day and presents what it found: "I found these potential tasks, want me to add them?" I confirm, adjust, or reject. The ones I confirm get routed to the right section of a unified task list with priority, time estimate, due date, and a values-based "why." That last part came from a therapy session. Every task gets a reason tied to a personal value, not a technical justification. "Close the bank account" isn't motivated by "close the bank account." It's motivated by "reduce mental overhead, be a good partner." When I'm staring at a task I've been dodging for three weeks, the value is what gets me to do it.

Completed tasks get moved to a completed section, not deleted. History matters. Patterns matter. Being able to look back and see that I consistently avoid the same category of task is useful information.

The shift is fundamental: task management becomes a byproduct of thinking out loud, not a separate administrative act. For an ADHD brain that will never reliably open a task app and type things in, that's everything.

## People Tracking

There's a `People/` folder in the vault with a file for everyone I interact with regularly. When someone gets tagged in a daily note or a meeting transcript, Claude checks if they have a file. If not, it creates one with their name, role, what they work on, and context from the interaction. If they already have a file, it adds a dated entry under an Interactions section with a summary of what we discussed.

Over time, each person's file becomes a running log of every meeting and conversation. Remember Harish from the voice memo example? Here's what his people file looks like:

```markdown
# Harish Arora

## Role
- Principal Product Manager covering CMX/STX

## Products
- CMX — context memory storage platform announced at GTC 2026

## Context
- Introduction via my manager
- Key contact for CMX product information and proprietary content

## Interactions
- 2026-03-30: Manager mentioned he'd introduce me to Harish
- 2026-03-31: Noted to reach out and schedule something
- 2026-04-06: First 1:1 call. Clarified CMX vs STX distinction, explained KV cache storage acceleration model, discussed partner execution timeline, benchmarks/lab coordination as my role.
- 2026-04-20: Led the partner biweekly sync. Introduced a teammate to present on hardware capabilities. Pushed for async progress with partner via Slack rather than limiting work to biweekly syncs.
```

I didn't write any of that. Claude built the file when Harish first came up in a daily note, and it's been adding interaction entries every time his name surfaces in a meeting or conversation. Before I hop on a call with someone I haven't talked to in a few weeks, I can ask Claude for a summary and get caught up in seconds. It's relationship CRM without the CRM, and without me ever entering a single contact record.

## What This Isn't

I want to be clear about what I'm not claiming here.

This is not automation for its own sake. Every workflow exists because I hit friction doing it manually and stopped doing it. As I said earlier in the post, I have tried every system that exists out there and nothing stuck, because the missing piece was always about removing friction. The check-in exists because I wouldn't journal. The voice memo pipeline exists because I wouldn't go back and listen to recordings. The task extraction exists because I wouldn't open a task app. Each one solves a specific failure mode of my brain, not a theoretical inefficiency.

This is not replacing my thinking. Claude is a scribe, an organizer, a researcher. The thinking is mine. The opinions are mine. The values are mine. When Claude drafts my accomplishments, I edit them. When it proposes tasks, I confirm or reject them. When it writes my daily note, it's writing my words, cleaned up and formatted. The intelligence in the system is the structure. The content is human.

This is not magic. It took weeks of iteration to get here. The check-in flow has been rewritten half a dozen times. The voice memo pipeline had bugs that re-processed already-transcribed memos and a new infuriating bug due to how iCloud stores data making it impossible for Claude Code to directly access the files from Voice Memos. The CLAUDE.md has sections that drifted and needed correction. This is engineering, not vibes. It works because I treated it like building a system, not installing an app.

And this is not for everyone. You need to be comfortable in a terminal. You need to think in markdown. You need to be willing to talk to a CLI tool like it's a person sitting next to you and not feel weird about it. If that's not you, that's fine. But if you've tried every productivity system and none of them stuck because they all assumed your brain works in a way it doesn't, maybe stop looking for a better app and start looking for a better collaborator.

## What It Changed

I journal every weekday now. I never did before. Not once in my life have I been able to sustain a journaling habit. Now I have 30+ days of daily notes with morning planning, afternoon reflections, biomarker data, linked notes, and extracted accomplishments. I didn't write any of them by hand. I talked, and they exist.

I have a running, linked, searchable record of my professional and personal development. Every meeting, every voice memo, every idea, every task, every person I've interacted with (within reason). Connected through wiki-links and tags. Queryable. It's not a pile of notes. It's a knowledge graph that grows every day.

Voice memos that would have disappeared into my phone become permanent knowledge. 76 old memos that sat untouched for months and years are now structured, filed, and connected to current work. Ideas I had in 2024 are linked to projects I'm doing in 2026.

Tasks don't fall through cracks. Not because I became more disciplined, but because the system doesn't require discipline. It requires conversation, and conversation is the one thing my brain does reliably.

I spend close to zero time using Obsidian itself. I open it on mobile occasionally to reference something. The vault is maintained entirely through Claude Code in the terminal. The app is the viewer. Claude is the operator.

At the end of each month, Claude generates a Month in Review with biomarker charts, top accomplishments, training coverage, notable deliverables, and a synopsis of procrastination and perfectionism patterns. The first one is happening today, actually. It hasn't happened yet as I write this, but I'm hoping it becomes the thing that lets me look back at a full month and see what went well, where I can improve, and whether the trajectory is pointing in the right direction. Self-awareness built from data I never manually tracked.

There's something deeper happening here that I want to name explicitly. The vault compounds. Every week, Claude has more context, makes better connections, writes more relevant briefs. A check-in on day 30 is fundamentally different from a check-in on day 1 because there are 29 days of history informing it. But it goes beyond just "more context." When you have a source of truth like this, Claude becomes more powerful, more accurate, and hallucinates far less. You stop fighting context window limits and weird AI memory quirks. You stop thinking about the machine entirely and start thinking about what you actually want to accomplish. The tool disappears.

I'm someone who doesn't want to pay attention to the details of how administrative things get done. I don't care about them. I don't like them. I don't think about them. But I need them. I don't care about those details so I *can* care about the details of the products of my work. Claude fills that gap. It handles the filing, the formatting, the tracking, the remembering, so my attention goes where it's supposed to go.

My wife has even noticed. Things at home are getting done more efficiently. I'm not actively using this setup in the evenings or on weekends, but I still have a better feel for what I've been forgetting or putting off because I've reviewed the personal items during my daily and weekly check-ins. The system doesn't just make me better at work. It makes me more present everywhere else because less is falling through the cracks.

If you have a neuro-spicy brain like mine, have struggled to find a note-taking and task management solution that actually sticks, and love the idea of having all the context for everything available at your fingertips, give this workflow a shot. Stop looking for an app. Start looking for a collaborator.

*Next up: I've been using the same approach to prepare for a homebrew D&D campaign. World-building, session prep, NPC tracking, and player arc management, all through Claude Code and Obsidian. That's a whole other post coming soon.*
