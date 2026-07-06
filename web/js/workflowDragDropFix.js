import { app } from "../../../scripts/app.js";

const WORKFLOW_EXTENSIONS = new Set(["json", "png", "webp", "jpg", "jpeg", "svg"]);
const HANDLED_MARK = "__zfWorkflowDropHandled";

function getFileExtension(file) {
	const name = file?.name || "";
	const index = name.lastIndexOf(".");
	return index >= 0 ? name.slice(index + 1).toLowerCase() : "";
}

function hasWorkflowFile(event) {
	const files = Array.from(event.dataTransfer?.files || []);
	return files.some((file) => WORKFLOW_EXTENSIONS.has(getFileExtension(file)));
}

function hasFileDrag(event) {
	const dataTransfer = event.dataTransfer;
	if (!dataTransfer) return false;
	if (Array.from(dataTransfer.items || []).some((item) => item.kind === "file")) return true;
	return Array.from(dataTransfer.types || []).includes("Files");
}

function isLocalDropTarget(event) {
	return Boolean(
		event.target?.closest?.(".comfy-multiline-input, input, textarea, select, button, .p-dialog, .litecontextmenu"),
	);
}

async function handleWorkflowDrop(event) {
	if (event[HANDLED_MARK]) return;
	if (isLocalDropTarget(event) || !hasWorkflowFile(event)) return;

	event[HANDLED_MARK] = true;
	const files = Array.from(event.dataTransfer?.files || []).filter((file) =>
		WORKFLOW_EXTENSIONS.has(getFileExtension(file)),
	);
	if (!files.length) return;

	event.preventDefault();
	event.stopPropagation();
	event.stopImmediatePropagation?.();

	for (const file of files) {
		await app.handleFile(file, "file_drop", { deferWarnings: true });
	}
}

app.registerExtension({
	name: "ZF.Helper.WorkflowDragDropFix",
	setup() {
		const handleDragOver = (event) => {
			if (isLocalDropTarget(event) || !hasFileDrag(event)) return;
			event.preventDefault();
			event.dataTransfer.dropEffect = "copy";
		};

		for (const target of [window, document, document.body].filter(Boolean)) {
			target.addEventListener("dragover", handleDragOver, true);
			target.addEventListener("drop", handleWorkflowDrop, true);
		}

		console.info("[ZF.Helper.WorkflowDragDropFix] workflow drag/drop fallback enabled");
	},
});
