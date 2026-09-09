export function shouldContinuePolling(status: string): boolean {
  return status === "pending" || status === "running";
}
