<template>
  <v-app id="inspire">
    <Navbar></Navbar>
    <v-main>
      <v-container
        class="fill-height"
        fluid
      >
        <v-row
          align="center"
          justify="center"
        >
        <router-view></router-view>
        </v-row>
      </v-container>
    </v-main>

    <v-footer app>
      <span>&copy; {{ new Date().getFullYear() }}</span>
    </v-footer>
  </v-app>
</template>

<script>
  import Navbar from './components/Navbar'
  export default {
    props: {
      source: String,
    },
    components:{
      Navbar
    },
    data: () => ({
  
    }),
    created () {
      this.$vuetify.theme.dark = true
    },
  }
</script>