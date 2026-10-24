/**
 * StabilizerToggle - Toggle component for prediction stabilization
 */

interface StabilizerToggleProps {
  enabled: boolean;
  onToggle: () => void;
}

export function StabilizerToggle({ enabled, onToggle }: StabilizerToggleProps) {
  const handleClick = (e: React.MouseEvent<HTMLInputElement>) => {
    e.preventDefault();
    onToggle();
  };

  return (
    <section className="section-style toggle-section">
      <div className="toggle-style">
        <label className="label-style">
          <input
            type="checkbox"
            checked={enabled}
            onChange={() => {}}
            className="input-style"
            aria-label="Enable prediction stabilization"
            onClick={handleClick}
          />
          <span>Enable Stabilization</span>
          <span className={`badge-style ${enabled ? 'badge-on' : 'badge-off'}`}>
            {enabled ? 'ON' : 'OFF'}
          </span>
        </label>
        <span className="note-style">
          {'Requires 3 consistent frames at >=70% confidence'}
        </span>
      </div>
    </section>
  );
}