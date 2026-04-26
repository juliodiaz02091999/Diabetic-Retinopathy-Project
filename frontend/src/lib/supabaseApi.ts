import { supabase, User, Patient } from './supabase'

// Auth API usando Supabase Auth
export const supabaseAuthAPI = {
  // Registro de usuario
  register: async (userData: {
    username: string
    name: string
    password: string
    email: string
  }) => {
    const { data, error } = await supabase.auth.signUp({
      email: userData.email,
      password: userData.password,
      options: {
        data: {
          username: userData.username,
          name: userData.name,
        }
      }
    })
    
    if (error) throw error
    
    // Comentado temporalmente para evitar errores de base de datos
    // Crear perfil de usuario en la tabla profiles
    // if (data.user) {
    //   const { error: profileError } = await supabase
    //     .from('profiles')
    //     .insert({
    //       id: data.user.id,
    //       username: userData.username,
    //       name: userData.name,
    //       email: userData.email
    //     })
    //   
    //   if (profileError) throw profileError
    // }
    
    return data
  },

  // Login
  login: async (credentials: { email: string; password: string }) => {
    const { data, error } = await supabase.auth.signInWithPassword({
      email: credentials.email,
      password: credentials.password,
    })
    
    if (error) throw error
    return data
  },

  // Logout
  logout: async () => {
    const { error } = await supabase.auth.signOut()
    if (error) throw error
  },

  // Obtener usuario actual
  getCurrentUser: async () => {
    const { data: { user }, error } = await supabase.auth.getUser()
    if (error) throw error
    return user
  },

  // Escuchar cambios en la autenticación
  onAuthStateChange: (callback: (event: string, session: any) => void) => {
    return supabase.auth.onAuthStateChange(callback)
  }
}

// Patient API usando Supabase
export const supabasePatientAPI = {
  // Crear paciente
  create: async (patientData: {
    name: string
    age: number
    gender: string
    contact_info: string
  }) => {
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) throw new Error('Usuario no autenticado')

    const { data, error } = await supabase
      .from('patients')
      .insert({
        user_id: user.id,
        ...patientData
      })
      .select()
      .single()

    if (error) throw error
    return data
  },

  // Obtener todos los pacientes del usuario actual
  getMyPatient: async () => {
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) throw new Error('Usuario no autenticado')

    const { data, error } = await supabase
      .from('patients')
      .select('*')
      .eq('user_id', user.id)
      .order('created_at', { ascending: false })

    if (error) throw error
    return data ?? []
  },

  // Actualizar paciente
  update: async (updates: Partial<Patient>) => {
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) throw new Error('Usuario no autenticado')

    const { data, error } = await supabase
      .from('patients')
      .update(updates)
      .eq('user_id', user.id)
      .select()
      .single()

    if (error) throw error
    return data
  }
}

// Prediction API usando Supabase
export const supabasePredictionAPI = {
  // Guardar predicción
  save: async (predictionData: {
    patient_id: string
    prediction_class: string
    confidence_score: number
    model_used?: string
  }) => {
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) throw new Error('Usuario no autenticado')

    const { data, error } = await supabase
      .from('predictions')
      .insert({
        patient_id: predictionData.patient_id,
        prediction_date: new Date().toISOString(),
        prediction_class: predictionData.prediction_class,
        confidence_score: predictionData.confidence_score / 100, // schema stores 0-1
        model_used: predictionData.model_used || 'Current Model'
      })
      .select()
      .single()

    if (error) throw error
    return data
  },

  // Obtener todas las predicciones del usuario
  getAll: async () => {
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) throw new Error('Usuario no autenticado')

    const { data, error } = await supabase
      .from('predictions')
      .select(`
        *,
        patients!inner(user_id)
      `)
      .eq('patients.user_id', user.id)
      .order('created_at', { ascending: false })

    if (error) throw error
    return data
  },

  // Obtener predicciones por paciente
  getByPatient: async (patientId: string) => {
    const { data, error } = await supabase
      .from('predictions')
      .select('*')
      .eq('patient_id', patientId)
      .order('created_at', { ascending: false })

    if (error) throw error
    return data
  }
}

// Profile API para manejar perfiles de usuario
export const supabaseProfileAPI = {
  // Obtener perfil del usuario actual
  getProfile: async () => {
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) throw new Error('Usuario no autenticado')

    const { data, error } = await supabase
      .from('profiles')
      .select('*')
      .eq('id', user.id)
      .single()

    if (error) throw error
    return data
  },

  // Actualizar perfil
  updateProfile: async (updates: Partial<User>) => {
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) throw new Error('Usuario no autenticado')

    const { data, error } = await supabase
      .from('profiles')
      .update(updates)
      .eq('id', user.id)
      .select()
      .single()

    if (error) throw error
    return data
  }
}

// Función para obtener el token de sesión (útil para llamadas al backend de ML)
export const getSessionToken = async () => {
  const { data: { session } } = await supabase.auth.getSession()
  return session?.access_token
} 