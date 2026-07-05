import { app } from "../../../scripts/app.js";

const EXTENSION_NAME = "ZF.Helper.ComboFolderMenus";
const SETTING_ID = "ZF.Helper.ComboFolderMenus.Enabled";
const PATCH_MARK = Symbol.for("zf.helper.comboFolderMenus.patched");
const FOLDER_MARK = "__zfComboFolderMenuFolder";
const VALUE_MARK = "__zfComboFolderMenuValue";
const SKIPPED_NODE_CLASSES = new Set(["CheckpointLoader|pysssss", "LoraLoader|pysssss"]);

function installStyles() {
	const styleId = "zf-helper-combo-folder-menu-style";
	if (document.getElementById(styleId)) return;

	const style = document.createElement("style");
	style.id = styleId;
	style.textContent = `
		.litecontextmenu .litemenu-entry.zf-combo-folder::before {
			content: "\\1F4C1";
			display: inline-block;
			font-size: 0.9em;
			margin-right: 6px;
			transform: translateY(-1px);
		}
		.litecontextmenu .litemenu-entry.zf-combo-leaf {
			white-space: nowrap;
		}
	`;
	document.head.appendChild(style);
}

function addSetting() {
	app.ui?.settings?.addSetting?.({
		id: SETTING_ID,
		name: "ZF Helper: group path-like combo menus",
		type: "boolean",
		defaultValue: true,
	});
}

function settingEnabled() {
	const value = app.ui?.settings?.getSettingValue?.(SETTING_ID);
	return value !== false;
}

function isPathLikeCombo(values, options) {
	if (!settingEnabled()) return false;
	if (options?.className !== "dark") return false;
	if (!Array.isArray(values) || values.length <= 4) return false;
	if (!values.every((value) => typeof value === "string")) return false;

	const currentNodeClass = app.canvas?.current_node?.comfyClass;
	if (SKIPPED_NODE_CLASSES.has(currentNodeClass)) return false;

	let pathLikeCount = 0;
	for (const value of values) {
		if (splitPath(value).length > 1) pathLikeCount++;
	}

	return pathLikeCount >= 2;
}

function splitPath(value) {
	return String(value).split(/[\\/]+/).filter(Boolean);
}

function createTreeNode() {
	return {
		folders: new Map(),
		leaves: [],
	};
}

function insertValue(root, value) {
	const parts = splitPath(value);
	if (parts.length <= 1) {
		root.leaves.push({ label: String(value), prefix: "", value });
		return;
	}

	let current = root;
	for (const folder of parts.slice(0, -1)) {
		if (!current.folders.has(folder)) current.folders.set(folder, createTreeNode());
		current = current.folders.get(folder);
	}

	current.leaves.push({
		label: parts[parts.length - 1],
		prefix: `${parts.slice(0, -1).join("/")}/`,
		value,
	});
}

function buildMenuOptions(node, callback) {
	const options = [];

	for (const [folderName, childNode] of node.folders.entries()) {
		options.push({
			content: folderName,
			className: "zf-combo-folder",
			has_submenu: true,
			[FOLDER_MARK]: true,
			submenu: {
				options: buildMenuOptions(childNode, callback),
				callback,
			},
		});
	}

	for (const leaf of node.leaves) {
		options.push({
			content: leaf.prefix
				? `<span style="display: none">${escapeHtml(leaf.prefix)}</span>${escapeHtml(leaf.label)}`
				: leaf.label,
			className: "zf-combo-leaf",
			[VALUE_MARK]: leaf.value,
		});
	}

	return options;
}

function escapeHtml(value) {
	return String(value)
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;")
		.replace(/'/g, "&#39;");
}

function groupValues(values, callback) {
	const root = createTreeNode();
	for (const value of values) insertValue(root, value);
	return buildMenuOptions(root, callback);
}

function patchContextMenu() {
	const ContextMenu = LiteGraph?.ContextMenu;
	if (!ContextMenu || ContextMenu[PATCH_MARK]) return;

	function FolderContextMenu(values, options = {}) {
		if (!isPathLikeCombo(values, options)) {
			return new ContextMenu(values, options);
		}

		const originalCallback = options.callback;
		const wrappedCallback = function (value, menuOptions, event, menu, extra) {
			if (value && typeof value === "object") {
				if (value[FOLDER_MARK]) return true;
				if (Object.prototype.hasOwnProperty.call(value, VALUE_MARK)) {
					return originalCallback?.call(this, value[VALUE_MARK], menuOptions, event, menu, extra);
				}
			}

			return originalCallback?.call(this, value, menuOptions, event, menu, extra);
		};

		const groupedValues = groupValues(values, wrappedCallback);
		const groupedOptions = {
			...options,
			callback: wrappedCallback,
			autoopen: true,
		};

		return new ContextMenu(groupedValues, groupedOptions);
	}

	FolderContextMenu.prototype = ContextMenu.prototype;
	Object.setPrototypeOf(FolderContextMenu, ContextMenu);
	FolderContextMenu[PATCH_MARK] = true;
	LiteGraph.ContextMenu = FolderContextMenu;
	console.info(`[${EXTENSION_NAME}] path-like combo menus are grouped by folder`);
}

app.registerExtension({
	name: EXTENSION_NAME,
	setup() {
		addSetting();
		installStyles();
		patchContextMenu();
	},
});
