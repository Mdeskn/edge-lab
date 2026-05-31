interface Props {
  connected: boolean;
  label: string;
}

export function StatusDot({ connected, label }: Props) {
  return (
    <span className="status-dot-wrap">
      <span className={`status-dot ${connected ? "is-connected" : "is-offline"}`} />
      {label}
    </span>
  );
}
