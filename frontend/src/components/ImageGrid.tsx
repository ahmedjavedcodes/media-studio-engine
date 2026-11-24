import { imageUrl, type ImageOut } from "@/lib/api";

// Each image shows as soon as it is ready. The final list replaces the early one.
export default function ImageGrid({ images }: { images: ImageOut[] }) {
  if (images.length === 0) return null;
  return (
    <div className="grid grid-cols-2 gap-3">
      {images.map((img, i) => (
        <a key={img.id} href={imageUrl(img.url)} target="_blank" rel="noreferrer" className="block">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={imageUrl(img.url)}
            alt={`Result ${i + 1}`}
            width={img.width}
            height={img.height}
            className="h-auto w-full rounded-lg border border-zinc-300 dark:border-zinc-700"
          />
        </a>
      ))}
    </div>
  );
}
