import AsyncStorage from "@react-native-async-storage/async-storage";

const HISTORY_KEY = "scan_history_v1";
const MAX_ITEMS = 30;

export async function getHistory() {
  const raw = await AsyncStorage.getItem(HISTORY_KEY);
  if (!raw) {
    return [];
  }
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export async function addHistoryItem(item) {
  const history = await getHistory();
  const merged = [item, ...history.filter((h) => h.id !== item.id)].slice(0, MAX_ITEMS);
  await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(merged));
  return merged;
}
