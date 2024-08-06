<template>
  <div id="app">
    <app-header v-if="header_show"></app-header>
    <router-view v-on:header="header" v-on:content="content" v-on:footer="footer"></router-view>
    <app-footer v-if="footer_show"></app-footer>
  </div>
</template>

<script>
import Header from "./components/Header";
import Footer from "./components/Footer";

export default {
  name: 'App',
  data(){
    return {
      header_show:true,
      footer_show:true,
    }
  },
  components: {
    "app-header": Header,
    "app-footer": Footer,
  },
  methods:{
    //是否显示头部
    header:function (bool) {
      this.header_show = bool;
    },
    //是否显示底部
    footer:function (bool) {
      this.footer_show = bool;
    }
  }
}
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 0px;
}
</style>
