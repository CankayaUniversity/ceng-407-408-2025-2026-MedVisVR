import * as T from "@radix-ui/react-tooltip";

export const TooltipProvider = ({ children }) => (
  <T.Provider delayDuration={350} skipDelayDuration={100}>
    {children}
  </T.Provider>
);

export function Tip({ children, content, side = "top", sideOffset = 7 }) {
  if (!content) return children;
  return (
    <T.Root>
      <T.Trigger asChild>{children}</T.Trigger>
      <T.Portal>
        <T.Content className="tip-content" side={side} sideOffset={sideOffset}>
          {content}
          <T.Arrow className="tip-arrow" />
        </T.Content>
      </T.Portal>
    </T.Root>
  );
}
