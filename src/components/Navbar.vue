<template>
    <nav>
    <v-app-bar
      app
      clipped-left
    >
      <v-app-bar-nav-icon @click.stop="drawer = !drawer"></v-app-bar-nav-icon>
      <v-toolbar-title>Menu</v-toolbar-title>
    </v-app-bar>
    <v-navigation-drawer
      v-model="drawer"
      app
      clipped
    >
      <v-list dense>
        <v-list-item v-for="link in links" :key="link.text" router :to="link.route">
          <v-list-item-action>
            <v-icon>{{ link.icon }}</v-icon>
          </v-list-item-action>
          <v-list-item-content>
            <v-list-item-title>{{ link.text }}</v-list-item-title>
          </v-list-item-content>
        </v-list-item>
      </v-list>
    </v-navigation-drawer>
    
    </nav>
</template>
<script>
export default {
    name: 'Navbar',
    data(){
        return{
            drawer: false,
            links: [
                { icon:'mdi-view-dashboard', text: 'Dashboard', route:'/dashboard'},
                { icon:'mdi-home', text:'Home', route:'/'},
                { icon:'mdi-poll', text: 'Sentiment', route:'/sentiment'},
            ]
        }
    }
}
</script>