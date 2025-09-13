# Memory Chat

This document explains the key architectural choices, trade‑offs, and implementation details behind this repo’s minimal per‑user memory chat. It is meant to be a slightly detailed reference for future contributors.

## Goals and Non‑Goals
- My goal with this implementation was to be a quick and easy implementation. Some of the key decisions I took includes:

1. Using gemini-2.5-flash as the model through which you can chat (reason: The memory implementation is independent of this model and I wanted to keep it cheap, using my personal gemini api key.)
2. Using gemini-2.5-flash-lite as the model to generate memories in a give input (input defined later.) (reason: We can use any model for this, but Deshraj mentioned we need SLM, now we can potentially train our own model specifically designed for this task to ensure we can do this with smallest possible model. I used gemini-flash-lite because it is the cheapest available model in gemini. Potentially can use together/groq or any other platform to use even cheaper model.)
3. Using gemini embedding model to extract embeddings of memories, now ideally we should build a custom embedding model specifically tuned for the task of memory, we don't need a general purpose embedding model, reason to use this model was to stay in the ecosystem of google. 
4. Keep database local, to reduce the overhead of setting things up on GCP. Good tradeoff for the purpose of assignment. I used faiss to implement the vector NN search. 
5. I have implemented two ways to save memory, now I couldn't do any literature review on which approach gives best output and what other strategies exists.
   1. First approach I took was, to extract memories from each user message, one issue I noticed naturally in one of my test conversation was, sometimes I select some option from assistant output instead of giving detailed verbose input, example is I asked gemini list google offices in bangalore, and I said, I go to first office. Now with only user input for memory extraction, we won't get which office user is talking about.
   2. The other approach I am using by default right now is, at the end of each session, we will give full transcript to extract memories, and thus save the memories only at the end of each session. This reduces chances of duplicates which we get within same session, though this approach can create context related issues especially if we build our own model (with maybe 32k context length), and is not scalable. 
6. See [Memory Data Model and Metadata](#memory-data-model-and-metadata)
   1. For memory, I extract embeddings by combining type, key and value. This approach is also not scalable, but for this simple implementation it gave better results for deduplication. 
   2. One key thing is expires_at, I couldn't validate it properly, but idea was to have expiry dates for ephemeral memories, which are valid only until a certain time. 
7. Finally although deduplication and merging is a difficult and important engineering challenge, I implemented a simple version of it, we find the nearest neighbour to each new memory, and if the similarity is above certain threshold we update the current memory with new memory to keep the most recent state. Another simple check I added was using the key, in case key is exact match, it makes search much simpler, though the models are stochastic and I couldn't get a single exact key match even with same queries. 

Sorry the code is not super extendible/maintainable, we can improve the architecture. I tried to make whatever changes I could in limited time. 
Thanks for giving me an opportunity to make a quick weekend project.

## Memory Data Model and Metadata
- **Table `memories`:**
  - `id INTEGER PRIMARY KEY AUTOINCREMENT`
  - `created_at TEXT` 
  - `memory_type TEXT` (one of: `preference`, `profile`, `fact`, `other`)
  - `key TEXT` (snake_case, normalized)
  - `value TEXT` (<= 60 chars expected by prompt, but not enforced by DB)
  - `sha1 TEXT UNIQUE` (of `type|key|value` for exact dedupe)
  - `expires_at TEXT` (YYYY‑MM‑DD or NULL)
  - `seen_count INTEGER` and `last_seen_at TEXT` (merge tracking)
  - `active INTEGER` (soft‑deactivation on conflict)
- **Table `meta`:** Stores small key/values such as `embed_dim` for consistency checks.
- **Index file:** FAISS index persisted to `mem.index` and kept in sync with row lifecycle.
