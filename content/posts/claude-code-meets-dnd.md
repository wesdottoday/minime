---
title: "Claude Code Meets D&D"
date: 2026-05-06
draft: false
tags: ["ai", "dnd", "claude-code", "adhd", "obsidian"]
---

I built a playable D&D campaign in 3.5 hours. Forty-one files. A world with history, politics, religion, a magic system, custom mechanics, nine named NPCs with full character sheets, a GM secrets layer with revelations designed to unfold across months of play, and a complete Session 1 cold open ready to run at the table.

I didn't write any of it by hand. I talked.

If you read [Claude Code Meets Obsidian](https://wes.today/posts/claude-code-meets-obsidian/), you know the premise: Claude Code isn't a chatbot. It's an agent with filesystem access. It reads, writes, searches, and modifies files directly inside my Obsidian vault. It knows the conventions, the folder structure, the tagging system. It operates *inside* the knowledge base, not outside it.

That post was about life management — daily check-ins, voice memos, task extraction, people tracking. Same tool, same vault. This time the domain is D&D.

I've been playing tabletop RPGs for years. I've got a group of close friends, each with distinct playstyles I know well. I had a campaign concept rattling around in my head: fragments of tone, themes, a world that felt like something, a central mystery I couldn't stop thinking about. What I didn't have was any of it written down.

This is the ADHD part. The ideas are there. They're vivid and interconnected and I can *feel* the campaign I want to run. But the distance between having the vision and having a rich, interlinked, playable campaign in a vault is a gap that executive function is supposed to bridge. Mine doesn't. I'll start organizing, get three files in, pull a thread on something more interesting, and come back two weeks later to an abandoned skeleton.

Claude doesn't lose the thread. Claude doesn't get bored of the structural work. I know what the world sounds like, what it feels like, who sits at my table and what they need from the game. Claude turns that into files, structure, and consistency. I talk. The campaign is born.

Here's how that actually works.

## The Collaboration Pattern

I don't outline. I don't write design documents. I talk about the thing I want to build, and we build it in conversation.

This isn't a style choice. It's how my brain works. I almost always have a broad design in my head for whatever I'm building, whether that's software, a campaign, or just an idea I'm chasing. But I don't enjoy the administrative side. I don't want to sit down and write my ideas out in complete thoughts and polished sentences. So I don't.

The problem is obvious: I come back to whatever I jotted down two weeks later and the context is gone. Terse bullet points that made perfect sense in the moment are useless without the thinking that produced them. The ideas were good. The documentation wasn't.

Claude fills that gap. I talk in fragments and half-formed concepts. Claude turns them into something I'd actually want to read later. It asks the follow-up questions that flesh out the parts I skipped, and writes everything into structured files while we're still in the conversation. The days of needing a rock-solid spec before the AI can do anything useful are gone, if you take this collaborative approach. It works for me. It may not work for you.

The pattern that emerged is simple: I give Claude a concept, usually in a few sentences. Claude expands it within the bounds of what we've already established. Then we pull on threads. Every concept has loose ends, and instead of filing them away for later, we follow them. All of them. Then reset and move to the next thing.

This is exactly how I built the [LLMs: All the Rabbit Holes](https://wes.today/series/inference/) series. Follow a thread until it dead-ends or branches, then follow the branches. Sometimes a random thought pops in mid-conversation that has nothing to do with the current thread. That's fine. I add it inline when talking to Claude, it flags it as something to come back to, and keeps going. Nothing gets lost. My inputs are messy. The conversation is solid. The output is structured.

Here's a real exchange from the session:

> **Wes:** The faction is a group of people, no one knows who they are, no individual has ever been identified by the normal public as a member. Most faction members appear as a normal person: blacksmith, soldier, tax collector... They're the people who truly control the world, the region. Not the governments.
>
> **Claude:** Good. Shadow government with total deniability. That's strong for the campaign clock too — early on, the bounty hunters don't even know who's paying them. The money just appears through intermediaries. **How do they maintain control?** A shadow faction that runs the world needs a mechanism. Some options to react to, or throw out entirely:
> - **Economic control** — they own the debt, the trade routes, the supply chains. Governments do what they say because the money says so.
> - **Information control** — they know things others don't. Secrets, leverage, blackmail. Power through asymmetric knowledge.
> - **Positional control** — they ARE the blacksmith, the tax collector, the captain of the guard. Not puppeteers pulling strings from above — they're embedded in the machinery itself.
> - **Some combination** — positional embedding gives them eyes and hands everywhere, but the real power comes from something else.
>
> That last option feels like it fits what you described — the positioning is the delivery mechanism, but the *source* of their power is something deeper. What's your read?
>
> **Wes:** Positional control. They are located in all parts of life, throughout life. They are embedded everywhere and nowhere all at once.
>
> **Claude:** That's clean. No conspiracy headquarters, no dark tower — just a network of ordinary people who happen to collectively steer everything. The tax collector adjusts a levy. The blacksmith delays an order. The harbormaster loses paperwork. Individually mundane. Collectively, the world moves where they want it to.

Three sentences from me. Claude came back with a menu of options, a recommendation, and a question. I picked one word. Claude translated it into scenes a GM can run at the table.

This happened over and over for 3.5 hours. I'd say something like "the magic system should be low fantasy, rare, feared." Claude would come back with a unified source for all magic types, implications for how society treats spellcasters, and the question I should have been asking: "If magic is this rare, what does a cleric's existence do to the local power structure?"

That's the thing that makes this work. It's not autocomplete. Claude isn't finishing my sentences. It's reading back what I said and showing me what it means. Every concept I dropped got tested against the world's internal logic before it hit the vault. By the time something became a file, it had already survived at least one round of "okay, but what does that break?"

## The Build Sequence

The whole session ran about 3.5 hours. Looking back at the file creation timestamps, five distinct phases emerged. I didn't plan them. They're just the natural shape of how the conversation moved from abstract to concrete.

**Foundation (first 40 minutes).** The world itself. What is this place? How does magic work? What do people believe? Who's in control and how? This is where the big load-bearing decisions get made. The world is called Rhenn. Low fantasy, pre-industrial everywhere. Magic is rare and feared. Religion exists but serves a purpose the believers don't understand. A shadow faction runs everything through positional control. These are the decisions that everything else hangs on.

**Locations and factions (next 30 minutes).** Zooming in. Twelve southern regions, each with a distinct identity and power structure. The northern frontier and what makes it different. The border town where Session 1 starts. The faction's internal structure and how it operates. This is where the world goes from concept to geography. Claude built each region with enough detail that I could improvise a scene in any of them, but not so much detail that I'm locked in before the players touch it.

**Player content (next 40 minutes).** The stuff my players will actually see. Character creation guide with the campaign's pitch, tone, and rules. Three origin paths with private handouts for each. Custom mechanical traits for the two cultures. This is the shift from "what is this world" to "how do players enter it." Claude wrote the handouts in the voice of the world, not the voice of a rulebook.

**NPCs (next 25 minutes).** This is where templates paid off. I built an NPC template for each culture with Claude, then immediately started using them. I'd give 10-15 words of input. Claude would return a full character sheet with personality, voicing notes, backstory, what they know, what they don't know, and a combat card if they might fight. Nine NPCs in 25 minutes. Here's what that looked like:

> **Wes:** How about an NPC template prompt for Ice Blood and for Southerner. I provide the key details — job, backstory, town, situation — and you produce a short NPC character sheet. This should also produce combat cards if I state they'll be in combat.
>
> **Claude:** Two templates in ZZZTemplates/. The workflow: You say something like "Ice Blood, female, blacksmith, border town, fought in a territorial dispute that cost her an eye, wary of strangers but fair. Combat-ready." I fill the template — personality, voicing notes, backstory, what she knows, what she doesn't, and a full combat card. Want to test-drive it?
>
> **Wes:** Let's try the blacksmith. Male. Ice blood. In the border town. Has a super brutal backstory.

Fifteen words. Here's what came back:

```markdown
---
title: Korr, Son of Asha
tags:
  - dnd/the-long-run
  - dnd/npc
  - dnd/ice-blood
class: Fighter (Champion)
level: 5
hp: 49
ac: 14
str: 18
con: 17
cha: 6
---

# Korr, Son of Asha

**Race:** Human (Ice Blood)
**Age:** Unknown. Appears somewhere between 40 and 120.
**Location:** The Town at the End of the World
**Role:** Blacksmith

## Appearance

Built like a boulder someone taught to stand upright. Deep burn scars run from his left wrist to his collarbone. His left ear is gone — not cut clean, torn. Three fingers on his left hand are fused together from a bad heal.

## Voicing Notes

- Almost never speaks first. Waits. Watches.
- When he does speak, it's blunt and final. "No." "It's done." "Come back tomorrow."
- Physical tells: rolls his fused fingers when thinking. Stares at the forge when uncomfortable.
- The one thing that makes him talk freely: his craft. Ask about steel, heat, the forge — more words in five minutes than he normally uses in a week.

## What He Knows

- Everything about the town — he's been here for years.
- The mountain pass — conditions, timing, equipment needed.
- Metalwork — can identify weapons, armor quality, and where gear was forged.

## What He Doesn't Know

- [Redacted — campaign secrets]. He is exactly what he appears to be.
- His sister's fate. If the PCs ever surfaced information about a woman taken by northern raiders decades ago, it would be the one thing that could make Korr act outside his nature.

## Combat Card

**Warhammer:** +7 to hit, 1d8+4 bludgeoning (versatile 1d10+4)
**Extra Attack, Action Surge (1/short rest), Improved Critical (19-20)**

He fights the way he works — measured, efficient, final.
```

That's a character I can voice at the table immediately. The backstory gives me emotional hooks to pull on later. The "What He Doesn't Know" section means I'll never accidentally have him reveal something he shouldn't. The voicing notes mean I don't have to remember how he talks. It's all right there. The best part though comes when we're in session. If I need a NPC, I don't have to quickly come up with a character that has no depth. I just give Claude 10-15 words and I get a full dossier for the NPC.

**Session prep (final 25 minutes).** The campaign bible. The Session 1 cold open with beat-by-beat pacing, including branching decision trees and the supporting assets. After the cold open, the world is wide open to the players. The GM philosophy document. And the CLAUDE.md that teaches Claude how to operate as my co-pilot during actual play.

> **Wes:** Can you drop a CLAUDE.md file in Sessions that explains how to navigate the vault, where key information is, how to track damage in the frontmatter? You will be used in each session and then at the end, I'll take the recording and transcript and we'll parse them, then write a session log. 

That's when I stopped thinking about the campaign as a thing to build and started thinking about it as a thing to run. The CLAUDE.md isn't worldbuilding. It's an operating manual. How Claude should navigate the vault during a session, how to track combat, how to process a recording into a session log afterward. More on how all of that works in the follow-up post.

## Iterative Refinement

Not everything lands on the first pass. Sometimes Claude made assumptions we hadn't discussed, and I'd catch it:

> **Wes:** Hold up, for [NPC] you put this in there: "[Detail implying certain characters live for centuries]." Did we discuss that?
>
> **Claude:** No, we didn't. That's me making an assumption I shouldn't have. What we established is that [the organization] has existed for centuries — but through succession, not individual longevity. The question this raises: does [the mechanism] extend a person's lifespan? Two options: 1. Normal lifespans — the organization persists through succession. 2. Extended life — which deepens the [campaign-specific effect] and makes the oldest members barely human anymore. Either way, I need to fix that line. What's your call?
>
> **Wes:** Normal lifespans, otherwise it might expose them.
>
> **Claude:** Makes perfect sense — a 300-year-old blacksmith would raise questions.

No defensiveness. No "well, what I meant was..." It named the mistake, offered two options with implications for each, and moved on. That matters for pace. You can't build this much depth in 3.5 hours if every correction turns into a negotiation.

But the more interesting moments were when the refinement wasn't about fixing mistakes. It was about making good ideas better.

> **Wes:** This is hard because I like the idea of [the opposing force] having its own sentience. I like the duality and the idea that there are two intelligences using the humans as chess pieces. But I fear that if [it] uses the same kind of attunement over time with their members that it might feel railroading to the party.
>
> **Claude:** That's the right instinct. If [the opposing force] just does the same thing with a blue paint job, the PCs traded one puppet master for another. Marcus will see it, and the table will feel it. But there's a way to have [it] be sentient AND preserve player agency: make [it] operate through a fundamentally different mechanism. [One side] enchants. It makes you *feel* aligned. [The other side] clarifies. It doesn't align you — it sharpens you. It makes you *see things as they are.*
>
> **Wes:** I like that. A final piece there, unlike [one side's] network, [the other side] does not [campaign secret here]. What [the opposing] seers see in the world is a society held together with duct tape and popsicle sticks. [They] are driven to give humanity free will.
>
> **Claude:** That's the campaign's thesis statement right there. **That makes both sides right and both sides dangerous.** [One] is correct that removing control would cause collapse. [The other] is correct that a civilization that can't stand on its own isn't really a civilization.

I started with a fear: this thing I liked might break the game for my players. Claude didn't talk me out of the concern or tell me it would be fine. It validated the worry, named which player would spot the problem first, and proposed a solution that made the whole concept stronger. Then I added one more constraint on top of that, and Claude recognized it as the thesis of the entire campaign. That recognition mattered. It told me the idea had weight.

This happened constantly. I'd say "I like that, but..." and the "but" would reshape whatever we'd just built. Claude treated every one of those as a design input, not a rejection. The difference matters. A tool that takes correction well is useful. A collaborator that treats your corrections as creative direction is something else.

## The Vault

Everything Claude and I built lives in Obsidian. Not Google Docs. Not Notion. Not a shared drive. Markdown files in folders on my machine, synced via iCloud, with wiki-links connecting everything together.

This matters because Claude Code operates directly on the filesystem. When I say "build an NPC for the border town," Claude doesn't ask me to describe the town. It reads the town's file, pulls the existing NPCs, checks the faction presence, and builds a character that fits what's already there. The vault is context. Every file Claude writes can reference every other file Claude has written.

Here's the folder structure:

```
The Long Run/
├── GM Only/          ← secrets, faction internals, revelation layers
├── World/
│   ├── Lore/         ← cosmology, magic system, cultures, religion
│   ├── Locations/    ← towns, regions, maps
│   └── Factions/     ← faction structure and operations
├── NPCs/             ← organized by location, templated
├── PCs/              ← character creation guide, origin handouts
├── Mechanics/        ← homebrew rules, custom traits
├── Sessions/
│   ├── Prep/         ← session plans, combat encounters
│   └── Logs/         ← post-session canonical records
└── ZZZTemplates/     ← NPC templates (Ice Blood, Southerner)
```

Every file has YAML front matter with tags. NPCs are tagged by culture, location, and role. Locations link to the NPCs who live there. Mechanics files link to the lore that explains why those mechanics exist. The faction files link to the regions they control. It's a web, not a list.

The tagging matters because it's how Claude finds things fast during play. If I say "who lives in the border town?" Claude doesn't search through every file. It filters by `#dnd/the-long-run` and location. If I say "what does this NPC know about the faction?" Claude reads their "What They Know" and "What They Don't Know" sections and gives me a straight answer.

The templates matter because they enforce consistency. Every NPC has the same sections in the same order. Appearance. Personality. Voicing notes. Backstory. What they know. What they don't know. Combat card if relevant. Interactions log for tracking what happens during play. When I'm at the table and I need to voice a character I haven't touched in three sessions, the voicing notes are right there. I don't have to remember. The vault remembers.

The GM Only folder is where the real architecture lives. The cosmology, the faction's true nature, the mechanism behind the central mystery. The stuff that makes the campaign work but that players should never stumble across. Claude can read those files alongside the player-facing content, so when I ask a question during play, the answer accounts for secrets the players haven't discovered yet.

## Designing for Your Table

Most campaign prep is generic. You build a world, write some encounters, and hope it works for the people sitting across from you. That's fine for published modules. It's not how I want to run a homebrew.

I know my players. I've played with most of them for years. I know Marcus will probe every system for internal contradictions. I know Dani will check behind every door and under every rug looking for what's hidden. I know Kev just finished a multi-year run as GM and wants mechanical consistency. I know Jamie is the most rules-literate person at the table. I know Ty will charge headfirst into whatever looks like combat and deal with consequences later.

Claude knew all of this too, because I told it. And it designed for them.

> **Claude** *(designing the first NPC interrogation):* What the bounty hunter knows — for when they interrogate him, and they will, Dani especially: He's a novice. Took the contract because he needed money. He was given the job through an intermediary — never met whoever's paying. He doesn't know about the other PCs. He doesn't know why. He didn't ask.

That "Dani especially" is the detail that matters. Claude isn't building an interrogation scene for a generic party. It's building one for a party where a specific person is highly likely to interrogate the NPC, and it's making sure the answers are ready.

This showed up everywhere. The campaign's central mystery has layered discovery built in because Marcus will pull on contradictions until the system reveals itself. The magic system has a single unified source because Kev will want to understand why it works, not just what it does. The rules are played loose because Jamie is there to catch edge cases, which means I don't have to. Combat encounters reward bold action because Ty needs to be able to charge in without the game punishing him for playing his way.

None of this is revolutionary game design. It's just preparation that accounts for the actual humans who will play. The difference is that Claude held all of these player profiles in context while building every piece of the campaign. Every NPC, every encounter, every mechanic got filtered through "how will this table interact with it?" without me having to remind Claude each time.

## The GM Layer

I can't talk about this section in detail. My players might read this post.

What I can say is that the campaign has secrets. Not "the villain's plan" secrets. Structural secrets. The kind that reshape what the players think the campaign is about. There are layers to the world that look like one thing from the outside and are something else entirely underneath. The players will discover them through play, not because I decide it's time for a reveal.

Claude helped me build those secrets with enough internal consistency to survive a table full of people who enjoy pulling on threads. Marcus will probe the logic. Dani will look for what's hidden. If the architecture doesn't hold up, they'll find the seams. So the architecture has to hold up.

Here's what I can describe without spoilers:

**Discovery layers.** The campaign's central mystery isn't revealed in one moment. It's structured as a series of layers that the players peel back through investigation, conversation, and pattern recognition. Each layer changes what they think they know. Claude and I designed five distinct layers, each one recontextualizing the one before it. The first layer hits in Session 1. The last one is the climax of the campaign.

**Mechanics that serve the story.** There are homebrew mechanics in this campaign that look like simple gameplay systems on the surface. They do something at the table. But they also do something to the narrative that the players won't recognize until much later. When they do, the mechanic becomes a question, not a rule. That's intentional. Claude helped me build mechanics where the gameplay value and the narrative value are the same thing.

**A campaign clock nobody can see.** There's a pressure system built into the premise that tightens over time without me having to force it. Early sessions feel open. Later sessions feel urgent. The players won't notice the shift happening because it's organic to the world. Nothing is artificial. The squeeze comes from the world reacting to what the players do, not from a timer I set.

**Three valid endings.** The campaign has a destination, but the path and the conclusion are genuinely open. I built three endings that are all honest to the premise. One is triumphant and dangerous. One is tragic and safe. One is something the players would have to invent. Claude pressure-tested all three against the world's logic and the table's likely choices. None of them are the "right" ending. All of them cost something.

I know this section is vague. It has to be. But the point isn't the specifics. The point is that Claude helped me build a campaign with hidden architecture that has depth, internal consistency, and enough structural integrity to survive players who will actively try to break it. That's hard to do alone. It's a lot easier when your collaborator can hold the entire world in context while you're designing the trap doors.

## What This Actually Is

I didn't use Claude to write a campaign for me. I used Claude to have the conversation I needed to have in order to build one.

The starting point was mine. The world was in my head before I opened a terminal. The tone, the themes, the kind of campaign I wanted to run. But the mechanics, the secrets, the layered details, the NPC depth, the faction architecture? That's where it ended up, not where it started. The collaboration took half-formed instincts and turned them into something I couldn't have built alone. Not because I lack the ability, but because my brain doesn't work that way by itself. The follow-up questions Claude asked shaped the answers. The pressure-testing refined the ideas. The structure made them real.

That's what Claude brought. Not just capacity. Partnership. The ability to hold a growing world in context and keep building on it without losing threads, dropping details, or getting bored of the structural work. My ADHD brain generates. Claude organizes. The vault remembers.

We haven't played Session 1 yet. That's coming in a month or two after our current campaign wraps up. When it does, Claude will be at the table with me. Not as a player. As the system that tracks combat, voices NPCs on demand, checks continuity, and processes the recording afterward into a canonical session log that updates the vault. That's Part 2 of this series.

For now, the campaign exists. Forty-one files. A world with secrets worth discovering. A Session 1 cold open that I'm genuinely excited to run. Built in a single conversation, in one afternoon, by talking "out loud".

If you want to try this yourself, here's a starter prompt you can drop into Claude Code to bootstrap your own campaign vault:

~~~markdown
I want to build a homebrew D&D campaign using Claude Code and Obsidian. I'll describe the world, the tone, and my players. You'll help me flesh it out, pressure-test it, and organize everything into a structured vault.

## How we work together

- I talk in concepts and fragments. You expand them, ask follow-up questions, and show me what my ideas imply.
- When you propose something, give me options where it makes sense. Don't just pick one.
- Pressure-test every decision against my players (I'll describe them) and against the world's internal logic. If something breaks, say so.
- When I say "I like that, but..." treat the "but" as a design input, not a rejection. Build on it.
- If you make an assumption we haven't discussed, own it and ask.
- Write files to the vault as we go. Don't wait until the end. By the time we're done talking, the campaign should be playable.

## Vault setup

Create this folder structure in my Obsidian vault:

<Campaign Name>/
├── CLAUDE.md
├── GM Only/
│   └── Open Questions.md
├── World/
│   ├── Lore/
│   ├── Locations/
│   └── Factions/
├── NPCs/
├── PCs/
├── Mechanics/
├── Sessions/
│   ├── Prep/
│   └── Logs/
└── ZZZTemplates/

## CLAUDE.md

Write a CLAUDE.md at the campaign root with these operating modes:

1. **Worldbuilder** — help me design the world, NPCs, locations, factions, encounters. Cross-link aggressively with [[wiki-links]]. Tag consistently. Flag open design questions rather than inventing answers I haven't approved.
2. **Session prep** — before each session, read previous session logs, open plot threads, NPC interaction history. Surface callbacks and foreshadowing opportunities. Build a session framework.
3. **In-session co-pilot** — fast responses. Pull from existing vault notes. Don't invent contradictions. Track combat in frontmatter (HP, conditions). Voice NPCs using their voicing notes.
4. **Transcript processor** — after each session, process the recording/transcript into a canonical session log. Update NPC interactions, canonize improvised worldbuilding, flag continuity risks, identify foreshadowing opportunities.

Include a section in the CLAUDE.md for the campaign premise, player profiles, tone/style guidance, and session pacing philosophy once we've established them.

## Templates

Create two NPC templates in ZZZTemplates/ with these sections:
- Front matter (title, date, tags with campaign tag)
- Name, Race/Origin, Age, Location, Role
- Appearance (2-3 sentences)
- Personality (voice, demeanor, traits)
- Voicing Notes (3-4 bullets for quick GM reference at the table)
- Backstory (3-5 sentences, GM reference)
- What They Know (bullet list)
- What They Don't Know (bullet list — gaps, blindspots, misconceptions)
- Combat Card (if applicable: level/CR, HP, AC, stats, actions, tactics)
- Interactions (dated entries added as campaign progresses)

Adapt the templates for the cultures/origins we establish during worldbuilding.

## Tagging

All files get:
- `dnd` (top-level)
- `dnd/<campaign-name>` (campaign-specific, kebab-case)
- Additional subtopic tags: `dnd/npc`, `dnd/location`, `dnd/lore`, `dnd/mechanic`, `dnd/faction`, `dnd/gm-only`, etc.

GM-private files get `dnd/gm-only` and live in GM Only/.

## Open Questions

Create an Open Questions.md in GM Only/ as a running log of unresolved design decisions. When we hit a question we're not ready to answer, add it here instead of guessing. Include a "Resolved" section for decisions we've locked.

## Let's start

Before we begin worldbuilding, ask me:
1. What's the campaign called?
2. What's the tone? (e.g., grimdark, heroic, comedic, pulpy)
3. What system are we using and how strict?
4. Tell me about your players — names, playstyles, what they need from the game.
5. What's the starting concept? (Even a fragment is fine. We'll build from there.)

~~~
