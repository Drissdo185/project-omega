## 📊 Complete Data Flow Visualization
```
User Query: "What methodology did authors use?"
    ↓
┌─────────────────────────────────────────────┐
│  1. Context Processing (if >8 turns)        │
│     Summarize conversation history          │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  2. Query Reformulation                     │
│     Input: "What methodology..."            │
│     Output: {reformulated_query, reasoning} │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  3. Query Classification                    │
│     Output: {needs_documents: true, ...}    │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  4. Task Planning                           │
│     Creates 2 tasks:                        │
│     - Task 1: Find methodology section      │
│     - Task 2: Extract experimental details  │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  5. Adaptive Execution                      │
│                                             │
│  Task 1:                                    │
│    → Page Selection (vision on 10 pages)    │
│       Returns: [3, 4, 5]                    │
│    → Analysis (vision on pages 3-5)         │
│       Findings: "Two-stage approach..."     │
│                                             │
│  Task 2:                                    │
│    → Page Selection                         │
│       Returns: [5, 6]                       │
│    → Analysis                               │
│       Findings: "Parameters include..."     │
│                                             │
│  Re-evaluation: Enough info? → Yes          │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  6. Response Synthesis                      │
│     Combines all findings                   │
│     Output: Comprehensive answer            │
└─────────────────────────────────────────────┘
    ↓
QueryResult(
    answer="The authors used a two-stage methodology...",
    page_numbers=[3, 4, 5, 6],
    total_cost=0.0234
)