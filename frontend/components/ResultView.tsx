import type { RunResult } from "@/lib/api";
import ImageGrid from "./ImageGrid";
import QuestionsForm from "./QuestionsForm";

interface Props {
  result: RunResult;
  onAnswers: (answers: Record<string, string>) => void;
  onRetry: () => void;
}

const box = "rounded-lg border p-4 ";

/** One UI state per status. */
export default function ResultView({ result, onAnswers, onRetry }: Props) {
  switch (result.status) {
    case "done":
      return <ImageGrid images={result.images} />;

    case "done_with_issues":
      return (
        <div className="space-y-4">
          <ImageGrid images={result.images} />
          <div className={box + "border-amber-300 bg-amber-50"}>
            <p className="mb-1 font-medium text-amber-900">Some things to know</p>
            <ul className="list-inside list-disc text-sm text-amber-900">
              {result.issues.map((issue) => (
                <li key={issue}>{issue}</li>
              ))}
            </ul>
          </div>
        </div>
      );

    case "needs_input":
      return (
        <div className={box + "border-indigo-200 bg-white"}>
          <p className="mb-3 font-medium">A little more information is needed.</p>
          <QuestionsForm questions={result.questions} onSubmit={onAnswers} />
        </div>
      );

    case "rejected":
      // Deliberately neutral: no detail about why, and never any images.
      return (
        <div className={box + "border-neutral-300 bg-white"}>
          <p className="font-medium">This image could not be delivered.</p>
          <p className="text-sm text-neutral-600">Try describing something different.</p>
        </div>
      );

    case "cancelled":
      return (
        <div className="space-y-4">
          <div className={box + "border-neutral-300 bg-white"}>
            <p className="font-medium">Cancelled.</p>
            {result.images.length > 0 && (
              <p className="text-sm text-neutral-600">These images were already checked and are yours to keep.</p>
            )}
          </div>
          <ImageGrid images={result.images} />
        </div>
      );

    case "failed":
      return (
        <div className={box + "border-red-200 bg-red-50"}>
          <p className="mb-3 font-medium text-red-900">Something went wrong and the image could not be created.</p>
          <button onClick={onRetry} className="rounded-md bg-red-600 px-4 py-2 font-medium text-white">
            Try again
          </button>
        </div>
      );

    case "escalate":
      return (
        <div className={box + "border-neutral-300 bg-white"}>
          <p className="font-medium">This request needs more than one step, which is not available yet.</p>
        </div>
      );
  }
}
