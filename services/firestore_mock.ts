/**
 * Mock Firestore Service
 * Simulates data privacy and structure based on User IDs.
 */

// In-memory simulated database
const MOCKED_DB: Record<string, any> = {
  "/global/config": { maintenanceMode: false, minVersion: "1.0.0" }
};

/**
 * Simulates writing data to a user-protected path.
 * Path format: /users/{uid}/{collection}/{docId}
 */
export const mockWriteUserData = async (uid: string, docId: string, data: any) => {
  await new Promise(resolve => setTimeout(resolve, 300));

  if (!uid) throw new Error("Permission Denied: User must be authenticated to write.");

  const path = `/users/${uid}/data/${docId}`;
  MOCKED_DB[path] = {
    ...data,
    _meta: {
      createdBy: uid,
      createdAt: new Date().toISOString()
    }
  };

  return { path, success: true };
};

/**
 * Simulates reading data from a user-protected path.
 * Enforces that only the owner (uid) can access this specific path pattern in this mock.
 */
export const mockGetUserData = async (uid: string, docId: string) => {
  await new Promise(resolve => setTimeout(resolve, 300));

  if (!uid) throw new Error("Permission Denied: User must be authenticated to read.");

  const path = `/users/${uid}/data/${docId}`;
  const data = MOCKED_DB[path];

  if (!data) {
    return null; 
  }

  return data;
};
