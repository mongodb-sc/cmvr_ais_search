"use client";

import Image from "next/image";
import { FormEvent, useEffect, useState } from "react";
import { Badge, Variant as BadgeVariant } from "@leafygreen-ui/badge";
import { Banner } from "@leafygreen-ui/banner";
import { Button, Variant as ButtonVariant } from "@leafygreen-ui/button";
import { Icon } from "@leafygreen-ui/icon";
import LeafyGreenProvider from "@leafygreen-ui/leafygreen-provider";
import { Spinner } from "@leafygreen-ui/loading-indicator/spinner";
import { Tab, Tabs } from "@leafygreen-ui/tabs";
import { TextArea } from "@leafygreen-ui/text-area";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import styles from "./research-workspace.module.css";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

type ToolStep = {
  index: number;
  name: string;
  summary: string;
  input: Record<string, unknown>;
  result: Record<string, unknown>;
};

type RunMeta = {
  turns: number;
  toolCalls: number;
  stoppedReason: string;
};

type HistorySummary = {
  id: string;
  query: string;
  createdAt: string;
  toolCalls: number;
};

type HistoryDetail = HistorySummary & RunMeta & {
  answer: string;
  steps: ToolStep[];
};

type StreamEvent =
  | ({ type: "tool" } & ToolStep)
  | ({ type: "done"; answer: string } & RunMeta)
  | { type: "error"; message: string };

export default function ResearchWorkspace() {
  const [isMounted, setIsMounted] = useState(false);
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [steps, setSteps] = useState<ToolStep[]>([]);
  const [answer, setAnswer] = useState("");
  const [meta, setMeta] = useState<RunMeta | null>(null);
  const [error, setError] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [history, setHistory] = useState<HistorySummary[]>([]);
  const [historyError, setHistoryError] = useState("");
  const [loadingHistoryId, setLoadingHistoryId] = useState<string | null>(null);

  useEffect(() => {
    setIsMounted(true);
    void loadHistory();
  }, []);

  if (!isMounted) {
    return <div className={styles.loadingShell} aria-label="Loading regulatory workspace" />;
  }

  async function runResearch(event?: FormEvent) {
    event?.preventDefault();
    const cleanQuery = query.trim();
    if (!cleanQuery || isRunning) return;

    setSubmittedQuery(cleanQuery);
    setSteps([]);
    setAnswer("");
    setMeta(null);
    setError("");
    setIsRunning(true);

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: cleanQuery }),
      });
      if (!response.ok || !response.body) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.trim()) continue;
          const streamEvent = JSON.parse(line) as StreamEvent;
          if (streamEvent.type === "tool") {
            setSteps((current) => [...current, streamEvent]);
          } else if (streamEvent.type === "done") {
            setAnswer(streamEvent.answer || "No answer produced.");
            setMeta(streamEvent);
            await loadHistory();
          } else {
            throw new Error(streamEvent.message);
          }
        }
        if (done) break;
      }
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : "Research failed.");
    } finally {
      setIsRunning(false);
    }
  }

  function clearResearch() {
    setQuery("");
    setSubmittedQuery("");
    setSteps([]);
    setAnswer("");
    setMeta(null);
    setError("");
  }

  async function loadHistory() {
    try {
      const response = await fetch(`${API_URL}/api/history`);
      if (!response.ok) throw new Error(`History request failed with status ${response.status}`);
      setHistory(await response.json() as HistorySummary[]);
      setHistoryError("");
    } catch (caughtError) {
      setHistoryError(caughtError instanceof Error ? caughtError.message : "Could not load history.");
    }
  }

  async function loadHistoryEntry(entry: HistorySummary) {
    if (isRunning) return;
    setLoadingHistoryId(entry.id);
    setError("");
    try {
      const response = await fetch(`${API_URL}/api/history/${entry.id}`);
      if (!response.ok) throw new Error(`History request failed with status ${response.status}`);
      const detail = await response.json() as HistoryDetail;
      setQuery(detail.query);
      setSubmittedQuery(detail.query);
      setSteps(detail.steps);
      setAnswer(detail.answer);
      setMeta({
        turns: detail.turns,
        toolCalls: detail.toolCalls,
        stoppedReason: detail.stoppedReason,
      });
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : "Could not load this research.");
    } finally {
      setLoadingHistoryId(null);
    }
  }

  async function clearHistory() {
    if (isRunning || history.length === 0) return;
    setHistoryError("");
    try {
      const response = await fetch(`${API_URL}/api/history`, { method: "DELETE" });
      if (!response.ok) throw new Error(`Clear history failed with status ${response.status}`);
      setHistory([]);
    } catch (caughtError) {
      setHistoryError(caughtError instanceof Error ? caughtError.message : "Could not clear history.");
    }
  }

  return (
    <LeafyGreenProvider baseFontSize={14}>
      <div className={styles.shell}>
        <header className={styles.topbar}>
          <div className={styles.brand}>
            <span className={styles.brandMark} aria-hidden="true"><Icon glyph="Database" /></span>
            <div>
              <strong>Regulatory Intelligence</strong>
              <span>CMVR / AIS research workspace</span>
            </div>
          </div>
          <div className={styles.systemStatus}>
            <span className={styles.statusDot} aria-hidden="true" />
            <span>Agent online</span>
            <Badge variant={BadgeVariant.Green}>MongoDB Atlas</Badge>
          </div>
        </header>

        <main className={styles.main}>
          <Tabs aria-label="Workspace views">
            <Tab default name={<><Icon glyph="Sparkle" /> Research</>}>
              <div className={styles.workspace}>
                <aside className={styles.queryPanel}>
                  <div className={styles.panelHeading}>
                    <p className={styles.overline}>New investigation</p>
                    <h1>Find applicable tests</h1>
                    <p>Search CMVR rules first, follow linked AIS clauses, and return traceable requirements.</p>
                  </div>

                  <form onSubmit={runResearch} className={styles.queryForm}>
                    <TextArea
                      label="Vehicle or approval question"
                      description="Include the vehicle category and changed system when known."
                      placeholder="e.g. What braking tests apply to an M3 bus?"
                      value={query}
                      onChange={(event) => setQuery(event.target.value)}
                      rows={6}
                      disabled={isRunning}
                    />
                    <div className={styles.formActions}>
                      <Button
                        type="submit"
                        variant={ButtonVariant.Primary}
                        leftGlyph={<Icon glyph="MagnifyingGlass" />}
                        disabled={!query.trim() || isRunning}
                      >
                        {isRunning ? "Researching" : "Run research"}
                      </Button>
                      <Button type="button" onClick={clearResearch} disabled={isRunning && !submittedQuery}>
                        Clear
                      </Button>
                    </div>
                  </form>

                  <section className={styles.history} aria-labelledby="history-heading">
                    <div className={styles.historyHeader}>
                      <div>
                        <h3 id="history-heading">Research history</h3>
                        <span>Latest 5 investigations</span>
                      </div>
                      <button type="button" onClick={clearHistory} disabled={isRunning || history.length === 0}>
                        <Icon glyph="Trash" size="small" />
                        Clear all
                      </button>
                    </div>
                    {historyError && <p className={styles.historyError}>{historyError}</p>}
                    <div className={styles.historyList}>
                      {history.length === 0 && !historyError && (
                        <p className={styles.historyEmpty}>Completed research will appear here.</p>
                      )}
                      {history.map((entry) => (
                        <button
                          className={styles.historyCard}
                          key={entry.id}
                          type="button"
                          onClick={() => loadHistoryEntry(entry)}
                          disabled={isRunning || loadingHistoryId !== null}
                        >
                          <span className={styles.historyCardIcon} aria-hidden="true">
                            {loadingHistoryId === entry.id ? <Spinner size="small" /> : <Icon glyph="Clock" size="small" />}
                          </span>
                          <span className={styles.historyCardCopy}>
                            <strong>{entry.query}</strong>
                            <small>{formatHistoryDate(entry.createdAt)} · {entry.toolCalls} {entry.toolCalls === 1 ? "search" : "searches"}</small>
                          </span>
                          <Icon glyph="ChevronRight" size="small" />
                        </button>
                      ))}
                    </div>
                  </section>
                </aside>

                <section className={styles.resultsPanel} aria-live="polite">
                  <div className={styles.resultsHeader}>
                    <div>
                      <p className={styles.overline}>Research output</p>
                      <h2>{submittedQuery || "Ready for a question"}</h2>
                    </div>
                    {isRunning && <Badge variant={BadgeVariant.Blue}>In progress</Badge>}
                    {answer && !isRunning && <Badge variant={BadgeVariant.Green}>Complete</Badge>}
                  </div>

                  {error && <Banner variant="danger">{error}</Banner>}

                  {!submittedQuery && !error && (
                    <div className={styles.emptyState}>
                      <span className={styles.emptyIcon}><Icon glyph="MagnifyingGlass" size="xlarge" /></span>
                      <h2>Evidence, not guesswork</h2>
                      <p>Your answer will appear here with the CMVR and AIS search path kept visible.</p>
                    </div>
                  )}

                  {submittedQuery && (
                    <div className={styles.outputGrid}>
                      <details className={styles.trace} open>
                        <summary className={`${styles.sectionTitle} ${styles.traceSummary}`}>
                          <h3>Evidence trace</h3>
                          <span className={styles.description}>{steps.length} tool {steps.length === 1 ? "call" : "calls"}</span>
                          <Icon glyph="ChevronDown" size="small" />
                        </summary>
                        <div className={styles.traceContent}>
                          {steps.map((step) => <ToolTrace key={`${step.index}-${step.name}`} step={step} />)}
                          {isRunning && (
                            <div className={styles.pendingStep}>
                              <Spinner size="small" />
                              <div><strong>{steps.length ? "Evaluating evidence" : "Searching CMVR rules"}</strong><span>The agent is building a cited response.</span></div>
                            </div>
                          )}
                        </div>
                      </details>

                      <article className={styles.answer}>
                        <div className={styles.sectionTitle}>
                          <h3>Applicable requirements</h3>
                          {meta && <span className={styles.description}>{meta.turns} turns · {meta.toolCalls} searches</span>}
                        </div>
                        {answer ? (
                          <div className={styles.markdown}><ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown></div>
                        ) : (
                          <div className={styles.answerPending}><p>The cited answer will be assembled after the evidence search completes.</p></div>
                        )}
                      </article>
                    </div>
                  )}
                </section>
              </div>
            </Tab>

            <Tab name={<><Icon glyph="Diagram2" /> Architecture</>}>
              <ArchitectureView />
            </Tab>
          </Tabs>
        </main>
      </div>
    </LeafyGreenProvider>
  );
}

function formatHistoryDate(value: string) {
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function ToolTrace({ step }: { step: ToolStep }) {
  const isCmvr = step.name === "cmvr_search";
  const count = isCmvr
    ? Array.isArray(step.result.rules) ? step.result.rules.length : 0
    : Array.isArray(step.result.clauses) ? step.result.clauses.length : 0;

  return (
    <details className={styles.traceStep} open>
      <summary>
        <span className={styles.stepIndex}>{step.index}</span>
        <span className={styles.stepCopy}>
          <strong>{isCmvr ? "CMVR rule search" : "AIS clause search"}</strong>
          <span>{count} {isCmvr ? "rules" : "clauses"} returned</span>
        </span>
        <Badge variant={isCmvr ? BadgeVariant.Blue : BadgeVariant.Purple}>{step.name}</Badge>
        <Icon glyph="ChevronDown" size="small" />
      </summary>
      <div className={styles.stepDetails}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{step.summary}</ReactMarkdown>
      </div>
    </details>
  );
}

function ArchitectureView() {
  return (
    <section className={styles.architecture}>
      <div className={styles.archHeading}>
        <p className={styles.overline}>System map</p>
        <h1>Evidence retrieval architecture</h1>
        {/* <p>The browser streams progress from the Python agent while the existing retrieval and model services remain unchanged.</p> */}
      </div>
      <div className={styles.archDiagram}>
        <Image
          className={styles.archImage}
          src="/architecture.svg"
          alt="CMVR and AIS agentic test-finder architecture showing the agent loop, hybrid retrieval, MongoDB collections, and cited response flow"
          width={1720}
          height={620}
          priority
        />
      </div>
      <div className={styles.archNotes}>
        <div><Badge variant={BadgeVariant.Blue}>1</Badge><h3>CMVR first</h3><p>The agent retrieves controlling rules and collects referenced AIS codes.</p></div>
        <div><Badge variant={BadgeVariant.Purple}>2</Badge><h3>Constrained AIS search</h3><p>AIS codes become hard filters, keeping clause retrieval relevant and auditable.</p></div>
        <div><Badge variant={BadgeVariant.Green}>3</Badge><h3>Hybrid ranking</h3><p>Atlas fuses vector and lexical candidates before Voyage reranks the evidence.</p></div>
      </div>
    </section>
  );
}
