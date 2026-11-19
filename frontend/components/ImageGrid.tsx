import type { ImageResult } from "@/lib/api";

/** Images, best first. They load from URLs; the score sits on each. */
export default function ImageGrid({ images }: { images: ImageResult[] }) {
  if (images.length === 0) return null;
  return (
    <ul className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      {images.map((image, i) => (
        <li key={image.image_id} className="overflow-hidden rounded-lg border border-neutral-200 bg-white">
          <div className="relative">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={image.url} alt={`Generated image ${i + 1}`} className="aspect-square w-full object-cover" />
            {image.quality_score !== null && (
              <span className="absolute right-2 top-2 rounded-full bg-black/70 px-2 py-0.5 text-xs font-medium text-white">
                {image.quality_score}/10
              </span>
            )}
          </div>
          <div className="px-3 py-2 text-sm text-neutral-600">{i === 0 && images.length > 1 ? "Best" : `Image ${i + 1}`}</div>
        </li>
      ))}
    </ul>
  );
}
