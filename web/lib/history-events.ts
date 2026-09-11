export const HISTORY_INVALIDATED_EVENT = "research-gap:history-invalidated";

export function invalidateHistory() {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(HISTORY_INVALIDATED_EVENT));
  }
}
